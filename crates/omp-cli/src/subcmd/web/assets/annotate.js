"use strict";
// Image annotation editor: numbered pins with comments plus shapes
// (arrows, lines, rectangles, ellipses, freehand) with zoom/pan.
// OMP.annotate(att) edits `att.pins` ([{x, y, text}]) and `att.annotations`
// ([{type, ...normalized coords, color?}]); x/y in 0..1 of the original
// image. Save re-renders `att.data` as a PNG with everything burned in;
// `att.orig`/`att.origType` keep the clean image so annotations stay
// editable. Old attachments that only have `att.pins` open fine.
// OMP.pinNotes(images) is the text block sent alongside the message.
(function (OMP) {
  const { el } = OMP;
  const SVGNS = "http://www.w3.org/2000/svg";
  const DEFAULT_COLOR = "#ff2d55";

  /** Toolbar icons: inline SVG, 24x24 grid, 2px currentColor stroke. */
  const ICONS = {
    select: '<path d="M5 4l7 16 2-7 7-2z"/>',
    pin: '<path d="M12 21s-7-5.5-7-11a7 7 0 0 1 14 0c0 5.5-7 11-7 11z"/><circle cx="12" cy="10" r="2.5"/>',
    arrow: '<path d="M5 19L19 5"/><path d="M11 5h8v8"/>',
    line: '<path d="M4 20L20 4"/>',
    rect: '<rect x="4" y="6" width="16" height="12" rx="1"/>',
    ellipse: '<ellipse cx="12" cy="12" rx="8" ry="6"/>',
    freehand: '<path d="M4 20l1-4L16 5l3 3L8 19z"/><path d="M14 7l3 3"/>',
    pan: '<path d="M12 3v18M3 12h18"/><path d="M9 5l3-3 3 3M9 19l3 3 3-3M5 9l-3 3 3 3M19 9l3 3-3 3"/>',
    del: '<path d="M4 7h16M9 7V5h6v2M6 7l1 13h10l1-13"/><path d="M10 11v6M14 11v6"/>',
    undo: '<path d="M4 10h11a5 5 0 0 1 0 10h-3"/><path d="M8 6l-4 4 4 4"/>',
    fit: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
  };
  function icon(name) {
    const wrap = document.createElement("span");
    wrap.innerHTML =
      `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"` +
      ` stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`;
    return wrap.firstChild;
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }

  function svgEl(tag, attrs, ...children) {
    const n = document.createElementNS(SVGNS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    n.append(...children);
    return n;
  }

  /** Render pins + shapes onto the original image, as base64 PNG. */
  async function burn(orig, origType, pins, annotations) {
    const img = await loadImage(`data:${origType};base64,${orig}`);
    const c = document.createElement("canvas");
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    const g = c.getContext("2d");
    g.drawImage(img, 0, 0);
    const W = c.width, H = c.height;
    const sw = Math.max(2, Math.round(Math.min(W, H) / 300));

    // shapes first, pins on top
    for (const a of annotations) {
      const color = a.color || DEFAULT_COLOR;
      g.strokeStyle = color; g.fillStyle = color;
      g.lineWidth = sw; g.lineCap = "round"; g.lineJoin = "round";
      const px = (v) => v * W, py = (v) => v * H;
      if (a.type === "line" || a.type === "arrow") {
        g.beginPath(); g.moveTo(px(a.x1), py(a.y1)); g.lineTo(px(a.x2), py(a.y2)); g.stroke();
        if (a.type === "arrow") {
          const ax = px(a.x2), ay = py(a.y2), bx = px(a.x1), by = py(a.y1);
          let dx = ax - bx, dy = ay - by;
          const len = Math.hypot(dx, dy) || 1;
          dx /= len; dy /= len;
          const hl = sw * 4, hw = sw * 2.2;
          g.beginPath();
          g.moveTo(ax, ay);
          g.lineTo(ax - dx * hl - dy * hw, ay - dy * hl + dx * hw);
          g.lineTo(ax - dx * hl + dy * hw, ay - dy * hl - dx * hw);
          g.closePath(); g.fill();
        }
      } else if (a.type === "rect") {
        g.strokeRect(px(a.x), py(a.y), px(a.w), py(a.h));
      } else if (a.type === "ellipse") {
        g.beginPath();
        g.ellipse(px(a.cx), py(a.cy), Math.abs(px(a.rx)), Math.abs(py(a.ry)), 0, 0, Math.PI * 2);
        g.stroke();
      } else if (a.type === "freehand" && a.points?.length > 1) {
        g.beginPath();
        g.moveTo(px(a.points[0].x), py(a.points[0].y));
        for (const p of a.points.slice(1)) g.lineTo(px(p.x), py(p.y));
        g.stroke();
      }
    }

    const r = Math.max(11, Math.round(Math.min(W, H) / 40));
    g.textAlign = "center"; g.textBaseline = "middle";
    g.font = `bold ${Math.round(r * 1.15)}px system-ui, sans-serif`;
    pins.forEach((p, i) => {
      const x = p.x * W, y = p.y * H;
      g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2);
      g.fillStyle = "#ff2d55"; g.fill();
      g.lineWidth = Math.max(2, r / 5); g.strokeStyle = "#fff"; g.stroke();
      g.fillStyle = "#fff"; g.fillText(String(i + 1), x, y + 1);
    });
    return c.toDataURL("image/png").split(",")[1];
  }

  /** Open the annotation editor for one attachment; resolves true when saved. */
  OMP.annotate = function annotate(att) {
    const orig = att.orig || att.data;
    const origType = att.origType || att.mimeType;
    const pins = (att.pins || []).map((p) => ({ ...p }));
    const annotations = (att.annotations || []).map((a) => ({
      ...a, points: a.points ? a.points.map((p) => ({ ...p })) : undefined,
    }));

    return new Promise((resolve) => {
      // One-gesture flow: a fresh capture (no pins/shapes yet) opens in pin
      // mode; reopening an annotated image starts in select to avoid stray pins.
      let mode = (att.pins?.length || att.annotations?.length) ? "select" : "pin";
      let selected = null; // {kind:"pin",index} | {kind:"shape",index}
      let zoom = 1, fit = 1, pan = { x: 0, y: 0 };
      let natW = 1, natH = 1;
      const undoStack = [];

      const snapshot = () => undoStack.push(JSON.stringify({ pins, annotations }));
      const restore = (s) => {
        const st = JSON.parse(s);
        pins.splice(0, pins.length, ...st.pins);
        annotations.splice(0, annotations.length, ...st.annotations);
      };

      // ---------- DOM ----------
      const img = el("img.an-img", { src: `data:${origType};base64,${orig}`, draggable: false });
      const shapesG = svgEl("g", { class: "an-shapes" });
      const pinsG = svgEl("g", { class: "an-pins" });
      const svg = svgEl("svg", { class: "an-svg" }, shapesG, pinsG);
      const panbox = el("div.an-panbox", null, img, svg);
      const stage = el("div.an-stage", null, panbox);
      const list = el("div.an-list");
      const status = el("span.an-status", { "aria-live": "polite" });
      const zoomLabel = el("span.an-zoompct", { text: "100%" });

      const tools = [
        ["select", "Select / move"], ["pin", "Add pin"],
        ["arrow", "Arrow"], ["line", "Line"],
        ["rect", "Rectangle"], ["ellipse", "Ellipse"],
        ["freehand", "Freehand"], ["pan", "Pan"],
      ];
      const toolBtns = {};
      const toolbar = el("div.an-toolbar", { role: "toolbar" });
      for (const [m, title] of tools) {
        const b = el("button.an-tool", { title, "aria-label": title, "data-mode": m, onclick: () => setMode(m) });
        b.append(icon(m));
        toolBtns[m] = b;
        toolbar.append(b);
      }
      const colorInput = el("input.an-color", { type: "color", value: DEFAULT_COLOR, title: "Shape color" });
      toolbar.append(colorInput);
      const delBtn = el("button.an-tool", { title: "Delete selected (Del)", "aria-label": "Delete selected", onclick: () => deleteSelected() });
      delBtn.append(icon("del"));
      const undoBtn = el("button.an-tool", { title: "Undo (Ctrl+Z)", "aria-label": "Undo", onclick: () => undo() });
      undoBtn.append(icon("undo"));
      const fitBtn = el("button.an-tool", { title: "Fit / reset zoom", "aria-label": "Fit / reset zoom", onclick: () => { zoom = 1; fitStage(); } });
      fitBtn.append(icon("fit"));
      toolbar.append(delBtn, undoBtn, el("span.an-sep"),
        el("button.an-tool", { text: "−", title: "Zoom out", "aria-label": "Zoom out", onclick: () => setZoom(zoom / 1.25) }),
        zoomLabel,
        el("button.an-tool", { text: "+", title: "Zoom in", "aria-label": "Zoom in", onclick: () => setZoom(zoom * 1.25) }),
        fitBtn);

      const cancelBtn = el("button.an-cancel", { text: "Cancel", onclick: () => close(false) });
      const saveBtn = el("button.primary.an-save", { text: "Save annotations", title: "Ctrl+Enter", onclick: () => close(true) });
      const dialog = el("section.an-dialog", { role: "dialog", "aria-modal": "true", "aria-label": "Image annotation editor" },
        el("div.an-main", null, toolbar, stage),
        el("aside.an-side", null,
          el("div.an-head", null, el("strong", { text: "Annotations" }), status),
          el("div.an-help", { text: "Pick a tool and draw on the image. Pins get numbered comments. Wheel zooms, drag with middle button or Space to pan." }),
          list,
          el("div.an-actions", null, cancelBtn, saveBtn)));
      const ov = el("div.an-overlay", null, dialog);

      function setMode(m) {
        mode = m;
        for (const k in toolBtns) toolBtns[k].classList.toggle("an-active", k === m);
        stage.classList.toggle("an-panmode", m === "pan");
      }

      // ---------- geometry ----------
      const scale = () => fit * zoom;
      const applyTransform = () => {
        panbox.style.transform = `translate(${pan.x}px,${pan.y}px) scale(${scale()})`;
        zoomLabel.textContent = `${Math.round(scale() * 100)}%`;
        updateSizes();
      };
      const fitStage = () => {
        const r = stage.getBoundingClientRect();
        fit = Math.min((r.width - 8) / natW, (r.height - 8) / natH) || 1;
        pan.x = (r.width - natW * scale()) / 2;
        pan.y = (r.height - natH * scale()) / 2;
        applyTransform();
      };
      const setZoom = (z, cx, cy) => {
        const nz = Math.min(20, Math.max(0.05, z));
        if (cx != null) {
          // keep the point under the cursor stationary
          const r = stage.getBoundingClientRect();
          const k = nz / zoom;
          pan.x = cx - r.left - (cx - r.left - pan.x) * k;
          pan.y = cy - r.top - (cy - r.top - pan.y) * k;
        }
        zoom = nz;
        applyTransform();
      };
      /** Client coords -> normalized image coords (transform-aware). */
      const toNorm = (cx, cy) => {
        const r = img.getBoundingClientRect();
        return { x: (cx - r.left) / r.width, y: (cy - r.top) / r.height };
      };

      // ---------- sizes that stay constant on screen under zoom ----------
      const SCREEN_STROKE = 2.5, PIN_R = 13;
      function updateSizes() {
        const s = scale();
        svg.querySelectorAll("[data-sw]").forEach((n) => n.setAttribute("stroke-width", SCREEN_STROKE / s));
        svg.querySelectorAll("[data-pinr]").forEach((n) => {
          const r = PIN_R / s;
          n.querySelector("circle").setAttribute("r", r);
          const t = n.querySelector("text");
          t.setAttribute("font-size", r * 1.15);
          t.setAttribute("dy", r * 0.07);
        });
      }

      // ---------- rendering ----------
      function shapeNode(a, i) {
        const s = scale();
        const sw = SCREEN_STROKE / s;
        const sel = selected?.kind === "shape" && selected.index === i;
        const common = {
          "data-sw": "1", "data-i": i, "vector-effect": "non-scaling-stroke",
          stroke: a.color || DEFAULT_COLOR, "stroke-width": sw,
          fill: "none", "stroke-linecap": "round", "stroke-linejoin": "round",
          class: sel ? "an-shape an-sel" : "an-shape",
        };
        let n;
        if (a.type === "line" || a.type === "arrow") {
          n = svgEl("line", { ...common, x1: a.x1 * natW, y1: a.y1 * natH, x2: a.x2 * natW, y2: a.y2 * natH });
          if (a.type === "arrow") {
            const head = svgEl("path", { ...common, fill: a.color || DEFAULT_COLOR, class: common.class + " an-arrowhead", "data-sw": "" });
            n.__head = head;
          }
        } else if (a.type === "rect") {
          n = svgEl("rect", { ...common, x: a.x * natW, y: a.y * natH, width: a.w * natW, height: a.h * natH });
        } else if (a.type === "ellipse") {
          n = svgEl("ellipse", { ...common, cx: a.cx * natW, cy: a.cy * natH, rx: Math.abs(a.rx * natW), ry: Math.abs(a.ry * natH) });
        } else if (a.type === "freehand") {
          const d = a.points.map((p, j) => `${j ? "L" : "M"}${p.x * natW},${p.y * natH}`).join("");
          n = svgEl("path", { ...common, d: d || "M0,0" });
        } else n = svgEl("g");
        if (n.__head) n.__head.setAttribute("d", arrowHeadPath(a));
        return n;
      }
      function arrowHeadPath(a) {
        const s = scale();
        const ax = a.x2 * natW, ay = a.y2 * natH, bx = a.x1 * natW, by = a.y1 * natH;
        let dx = ax - bx, dy = ay - by;
        const len = Math.hypot(dx, dy) || 1;
        dx /= len; dy /= len;
        const hl = SCREEN_STROKE * 4 / s, hw = SCREEN_STROKE * 2.2 / s;
        return `M${ax},${ay}L${ax - dx * hl - dy * hw},${ay - dy * hl + dx * hw}L${ax - dx * hl + dy * hw},${ay - dy * hl - dx * hw}Z`;
      }

      function renderShapes() {
        shapesG.replaceChildren();
        annotations.forEach((a, i) => {
          const n = shapeNode(a, i);
          shapesG.append(n);
          if (n.__head) shapesG.append(n.__head);
        });
      }

      function renderPins(focus) {
        pinsG.replaceChildren(...pins.map((p, i) => {
          const g = svgEl("g", {
            "data-pinr": "1", "data-i": i, class: "an-pin",
            transform: `translate(${p.x * natW},${p.y * natH})`,
          }, svgEl("circle", { r: PIN_R, fill: "#ff2d55", stroke: "#fff", "stroke-width": 2.5, "vector-effect": "non-scaling-stroke" }),
             svgEl("text", { "text-anchor": "middle", fill: "#fff", "font-weight": "bold", "font-family": "system-ui, sans-serif" }, String(i + 1)));
          if (selected?.kind === "pin" && selected.index === i) g.classList.add("an-sel");
          return g;
        }));
        list.replaceChildren(...pins.map((p, i) => el("div.an-row", {
          class: selected?.kind === "pin" && selected.index === i ? "an-row an-row-sel" : "an-row",
          onclick: (e) => {
            // Interactive controls must not trigger the destructive row
            // re-render: clicking into (or typing in) the comment textarea
            // would otherwise replace it and drop focus/value. Selection is
            // applied in place by toggling classes only.
            selectPin(i);
          },
        },
          el("span.an-num", { text: String(i + 1) }),
          el("textarea", {
            rows: 2, placeholder: "What to change here…", text: p.text || "",
            oninput: (e) => { p.text = e.target.value; },
            // Never let a click/keystroke inside the field bubble into a
            // re-render or a pin deletion.
            onkeydown: (e) => e.stopPropagation(),
            onclick: (e) => e.stopPropagation(),
          }),
          el("button.an-del", { text: "×", title: "Remove pin", onclick: (e) => { e.stopPropagation(); snapshot(); pins.splice(i, 1); selected = null; render(); } }))));
        if (focus != null) list.querySelectorAll("textarea")[focus]?.focus();
        updateSizes();
      }

      /** Select a pin row without rebuilding the list (keeps focus/value). */
      function selectPin(i) {
        const already = selected?.kind === "pin" && selected.index === i;
        selected = { kind: "pin", index: i };
        if (!already) {
          list.querySelectorAll(".an-row").forEach((r, j) => r.classList.toggle("an-row-sel", j === i));
          pinsG.querySelectorAll(".an-pin").forEach((g, j) => g.classList.toggle("an-sel", j === i));
        }
      }

      function render(focus) {
        const n = pins.length + annotations.length;
        status.textContent = n ? `${n} item${n === 1 ? "" : "s"} · not saved` : "No annotations yet";
        renderShapes();
        renderPins(focus);
      }

      // ---------- editing ops ----------
      function deleteSelected() {
        if (!selected) return;
        snapshot();
        if (selected.kind === "pin") pins.splice(selected.index, 1);
        else annotations.splice(selected.index, 1);
        selected = null;
        render();
      }
      function undo() {
        const s = undoStack.pop();
        if (s == null) return;
        restore(s);
        selected = null;
        render();
      }

      function startPinDrag(e, i) {
        snapshot();
        const g = pinsG.children[i];
        g.setPointerCapture(e.pointerId);
        selected = { kind: "pin", index: i };
        const move = (m) => {
          const p = toNorm(m.clientX, m.clientY);
          pins[i].x = Math.min(1, Math.max(0, p.x));
          pins[i].y = Math.min(1, Math.max(0, p.y));
          g.setAttribute("transform", `translate(${pins[i].x * natW},${pins[i].y * natH})`);
        };
        const up = () => {
          g.removeEventListener("pointermove", move);
          render(i);
          list.querySelectorAll("textarea")[i]?.focus();
        };
        g.addEventListener("pointermove", move);
        g.addEventListener("pointerup", up, { once: true });
      }

      function startShapeDrag(e, i) {
        snapshot();
        const a = annotations[i];
        selected = { kind: "shape", index: i };
        const start = toNorm(e.clientX, e.clientY);
        const orig = JSON.parse(JSON.stringify(a));
        stage.setPointerCapture(e.pointerId);
        const apply = (dx, dy) => {
          const cp = JSON.parse(JSON.stringify(orig));
          if (cp.type === "freehand") cp.points.forEach((p) => { p.x += dx; p.y += dy; });
          else if (cp.type === "rect") { cp.x += dx; cp.y += dy; }
          else if (cp.type === "ellipse") { cp.cx += dx; cp.cy += dy; }
          else { cp.x1 += dx; cp.y1 += dy; cp.x2 += dx; cp.y2 += dy; }
          annotations[i] = cp;
          renderShapes();
          updateSizes();
        };
        const move = (m) => {
          const p = toNorm(m.clientX, m.clientY);
          apply(p.x - start.x, p.y - start.y);
        };
        const up = (m) => {
          stage.removeEventListener("pointermove", move);
          const p = toNorm(m.clientX, m.clientY);
          apply(p.x - start.x, p.y - start.y);
        };
        stage.addEventListener("pointermove", move);
        stage.addEventListener("pointerup", up, { once: true });
      }

      /** Create a shape from a finished drag; returns it if big enough. */
      function finishShape(type, start, cur, pts) {
        const color = colorInput.value;
        const min = 0.004;
        let a = null;
        if (type === "line" || type === "arrow") {
          if (Math.hypot(cur.x - start.x, cur.y - start.y) < min) return null;
          a = { type, x1: start.x, y1: start.y, x2: cur.x, y2: cur.y, color };
        } else if (type === "rect") {
          if (Math.abs(cur.x - start.x) < min || Math.abs(cur.y - start.y) < min) return null;
          a = { type, x: Math.min(start.x, cur.x), y: Math.min(start.y, cur.y), w: Math.abs(cur.x - start.x), h: Math.abs(cur.y - start.y), color };
        } else if (type === "ellipse") {
          if (Math.abs(cur.x - start.x) < min || Math.abs(cur.y - start.y) < min) return null;
          a = { type, cx: (start.x + cur.x) / 2, cy: (start.y + cur.y) / 2, rx: (cur.x - start.x) / 2, ry: (cur.y - start.y) / 2, color };
        } else if (type === "freehand") {
          if (!pts || pts.length < 2) return null;
          a = { type, points: pts, color };
        }
        return a;
      }

      // ---------- pointer handling on the stage ----------
      let drawing = null;
      stage.addEventListener("pointerdown", (e) => {
        if (e.button === 1 || (e.button === 0 && (mode === "pan" || e.getModifierState?.(" ") || e.shiftKey && mode === "select"))) {
          // pan
          e.preventDefault();
          stage.setPointerCapture(e.pointerId);
          const sx = e.clientX - pan.x, sy = e.clientY - pan.y;
          const move = (m) => { pan.x = m.clientX - sx; pan.y = m.clientY - sy; applyTransform(); };
          stage.addEventListener("pointermove", move);
          stage.addEventListener("pointerup", () => stage.removeEventListener("pointermove", move), { once: true });
          return;
        }
        if (e.button !== 0) return;
        const p = toNorm(e.clientX, e.clientY);
        if (p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1) {
          if (mode === "select") { selected = null; render(); }
          return;
        }
        if (mode === "select") {
          selected = null; render();
          return;
        }
        if (mode === "pin") {
          e.preventDefault(); // keep focus on the new pin's comment field
          snapshot();
          pins.push({ x: Math.min(1, Math.max(0, p.x)), y: Math.min(1, Math.max(0, p.y)), text: "" });
          render(pins.length - 1);
          return;
        }
        // drawing modes
        e.preventDefault();
        stage.setPointerCapture(e.pointerId);
        snapshot();
        const start = p;
        const pts = mode === "freehand" ? [p] : null;
        drawing = finishShape(mode, start, p, pts) || { type: mode, color: colorInput.value };
        const idx = annotations.push(drawing) - 1;
        const move = (m) => {
          const cur = toNorm(m.clientX, m.clientY);
          if (mode === "freehand") {
            pts.push(cur);
            annotations[idx] = finishShape(mode, start, cur, pts) || annotations[idx];
          } else {
            annotations[idx] = finishShape(mode, start, cur, pts) || { type: mode, color: colorInput.value };
          }
          drawing = annotations[idx];
          renderShapes(); updateSizes();
        };
        const up = (m) => {
          stage.removeEventListener("pointermove", move);
          const cur = toNorm(m.clientX, m.clientY);
          const done = finishShape(mode, start, cur, pts);
          if (done) { annotations[idx] = done; selected = { kind: "shape", index: idx }; }
          else annotations.splice(idx, 1);
          drawing = null;
          render();
        };
        stage.addEventListener("pointermove", move);
        stage.addEventListener("pointerup", up, { once: true });
      });

      svg.addEventListener("pointerdown", (e) => {
        if (mode !== "select" || e.button !== 0) return;
        const t = e.target.closest(".an-pin");
        if (t) { e.stopPropagation(); startPinDrag(e, Number(t.dataset.i)); return; }
        const s = e.target.closest(".an-shape");
        if (s && s.dataset.i != null && !s.classList.contains("an-arrowhead")) {
          e.stopPropagation();
          startShapeDrag(e, Number(s.dataset.i));
        }
      }, true);

      stage.addEventListener("wheel", (e) => {
        e.preventDefault();
        setZoom(zoom * (e.deltaY < 0 ? 1.12 : 1 / 1.12), e.clientX, e.clientY);
      }, { passive: false });

      // ---------- keyboard ----------
      let closed = false, saving = false;
      const onKey = (e) => {
        if (saving) return;
        const inField = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "");
        if (e.key === "Escape") { e.preventDefault(); close(false); }
        else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); close(true); }
        else if ((e.key === "Delete" || e.key === "Backspace") && !inField) { e.preventDefault(); deleteSelected(); }
        else if ((e.key === "z" || e.key === "Z") && (e.ctrlKey || e.metaKey) && !inField) { e.preventDefault(); undo(); }
      };
      const onResize = () => fitStage();
      const cleanup = () => {
        window.removeEventListener("keydown", onKey, true);
        window.removeEventListener("resize", onResize);
        ov.remove();
      };

      async function close(save) {
        if (closed || saving) return;
        if (!save) {
          closed = true; cleanup(); resolve(false); return;
        }
        saving = true;
        cancelBtn.disabled = true; saveBtn.disabled = true; saveBtn.textContent = "Saving…";
        status.textContent = "Rendering…";
        try {
          const has = pins.length || annotations.length;
          const data = has ? await burn(orig, origType, pins, annotations) : orig;
          // Commit only after the rendered attachment is ready. On failure the
          // editor and all unsaved annotations remain available for another try.
          att.pins = pins.map((p) => ({ ...p }));
          if (annotations.length) att.annotations = annotations.map((a) => ({ ...a, points: a.points?.map((p) => ({ ...p })) }));
          else delete att.annotations;
          if (has) {
            att.orig = orig; att.origType = origType;
            att.data = data; att.mimeType = "image/png";
          } else {
            att.data = orig; att.mimeType = origType;
            delete att.orig; delete att.origType;
          }
          closed = true; cleanup(); resolve(true);
        } catch (e) {
          saving = false;
          cancelBtn.disabled = false; saveBtn.disabled = false; saveBtn.textContent = "Save annotations";
          status.textContent = "Save failed · annotations preserved";
          OMP.toast("Could not save image annotations: " + (e.message || e), "error", 7000);
        }
      }

      // ---------- init ----------
      setMode("select");
      window.addEventListener("keydown", onKey, true);
      window.addEventListener("resize", onResize);
      document.body.append(ov);
      const init = () => {
        natW = img.naturalWidth || 1;
        natH = img.naturalHeight || 1;
        img.style.width = `${natW}px`;
        img.style.height = `${natH}px`;
        svg.setAttribute("viewBox", `0 0 ${natW} ${natH}`);
        svg.setAttribute("width", natW);
        svg.setAttribute("height", natH);
        fitStage();
        render();
      };
      if (img.complete && img.naturalWidth) init();
      else img.onload = init;
    });
  };

  /** Pin comments of all attachments, numbered like the burned-in markers. */
  OMP.pinNotes = function pinNotes(images) {
    const blocks = [];
    images.forEach((img, n) => {
      const label = images.length > 1 ? `image ${n + 1}` : "the image";
      if (!img.pins?.length) {
        if (img.caption) blocks.push(`Attached ${label}: ${img.caption}`);
        return;
      }
      const head = `Pins on ${label}${img.caption ? ` — ${img.caption}` : ""}:`;
      const lines = img.pins.map((p, i) =>
        `${i + 1}. (x ${Math.round(p.x * 100)}%, y ${Math.round(p.y * 100)}%) ${(p.text || "").trim() || "(no comment)"}`);
      blocks.push([head, ...lines].join("\n"));
    });
    return blocks.join("\n\n");
  };
})(window.OMP);
