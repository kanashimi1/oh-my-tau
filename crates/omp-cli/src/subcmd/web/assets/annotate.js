"use strict";
// Image pins: numbered points with comments on a composer attachment.
// OMP.annotate(att) edits `att.pins` ([{x, y, text}], x/y in 0..1 of the
// original image) and re-renders `att.data` as the original with the pins
// burned in; `att.orig` keeps the clean image so pins stay editable.
// OMP.pinNotes(images) is the text block sent alongside the message.
(function (OMP) {
  const { el } = OMP;

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }

  /** Original image with the pin markers drawn on top, as base64 PNG. */
  async function burnPins(orig, mimeType, pins) {
    const img = await loadImage(`data:${mimeType};base64,${orig}`);
    const c = document.createElement("canvas");
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    const g = c.getContext("2d");
    g.drawImage(img, 0, 0);
    const r = Math.max(11, Math.round(Math.min(c.width, c.height) / 40));
    g.textAlign = "center"; g.textBaseline = "middle";
    g.font = `bold ${Math.round(r * 1.15)}px system-ui, sans-serif`;
    pins.forEach((p, i) => {
      const x = p.x * c.width, y = p.y * c.height;
      g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2);
      g.fillStyle = "#ff2d55"; g.fill();
      g.lineWidth = Math.max(2, r / 5); g.strokeStyle = "#fff"; g.stroke();
      g.fillStyle = "#fff"; g.fillText(String(i + 1), x, y + 1);
    });
    return c.toDataURL("image/png").split(",")[1];
  }

  /** Open the pin editor for one attachment; resolves true when saved. */
  OMP.annotate = function annotate(att) {
    const orig = att.orig || att.data;
    const origType = att.origType || att.mimeType;
    const pins = (att.pins || []).map((p) => ({ ...p }));
    return new Promise((resolve) => {
      const img = el("img.pin-img", { src: `data:${origType};base64,${orig}`, draggable: false });
      const layer = el("div.pin-layer");
      const stage = el("div.pin-stage", null, img, layer);
      const list = el("div.pin-list");
      const ov = el("div.pin-overlay", null,
        stage,
        el("aside.pin-side", null,
          el("div.pin-help", { text: "Click the image to drop a numbered pin, then describe the fix. Drag a pin to move it." }),
          list,
          el("div.pin-actions", null,
            el("button", { text: "Cancel", onclick: () => close(false) }),
            el("button.primary", { text: "Done", title: "Ctrl+Enter", onclick: () => close(true) }))));

      // object-fit: contain letterboxes the image inside the stage
      const box = () => {
        const e = img.getBoundingClientRect();
        const k = Math.min(e.width / img.naturalWidth, e.height / img.naturalHeight) || 1;
        const w = img.naturalWidth * k, h = img.naturalHeight * k;
        return { left: e.left + (e.width - w) / 2, top: e.top + (e.height - h) / 2, w, h };
      };
      const toImage = (cx, cy) => {
        const b = box();
        return { x: Math.min(1, Math.max(0, (cx - b.left) / b.w)), y: Math.min(1, Math.max(0, (cy - b.top) / b.h)) };
      };

      function render(focus) {
        const b = box(), s = stage.getBoundingClientRect();
        layer.replaceChildren(...pins.map((p, i) => {
          const dot = el("div.pin-dot", { text: String(i + 1) });
          dot.style.left = `${b.left - s.left + p.x * b.w}px`;
          dot.style.top = `${b.top - s.top + p.y * b.h}px`;
          dot.addEventListener("pointerdown", (e) => {
            e.stopPropagation();
            dot.setPointerCapture(e.pointerId);
            const move = (m) => { Object.assign(p, toImage(m.clientX, m.clientY)); render(); };
            dot.addEventListener("pointermove", move);
            dot.addEventListener("pointerup", () => {
              dot.removeEventListener("pointermove", move);
              list.querySelectorAll("textarea")[i]?.focus();
            }, { once: true });
          });
          return dot;
        }));
        list.replaceChildren(...pins.map((p, i) => el("div.pin-row", null,
          el("span.pin-num", { text: String(i + 1) }),
          el("textarea", {
            rows: 2, placeholder: "What to change here…", value: p.text || "",
            oninput: (e) => { p.text = e.target.value; },
          }),
          el("button.pin-del", { text: "×", title: "Remove pin", onclick: () => { pins.splice(i, 1); render(); } }))));
        if (focus != null) list.querySelectorAll("textarea")[focus]?.focus();
      }

      stage.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        const b = box();
        if (e.clientX < b.left || e.clientX > b.left + b.w || e.clientY < b.top || e.clientY > b.top + b.h) return;
        pins.push({ ...toImage(e.clientX, e.clientY), text: "" });
        render(pins.length - 1);
      });
      const onKey = (e) => {
        if (e.key === "Escape") { e.preventDefault(); close(false); }
        else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); close(true); }
      };
      const onResize = () => render();

      async function close(save) {
        window.removeEventListener("keydown", onKey, true);
        window.removeEventListener("resize", onResize);
        ov.remove();
        if (!save) return resolve(false);
        att.pins = pins;
        if (pins.length) {
          att.orig = orig; att.origType = origType;
          att.data = await burnPins(orig, origType, pins);
          att.mimeType = "image/png";
        } else {
          att.data = orig; att.mimeType = origType;
          delete att.orig; delete att.origType;
        }
        resolve(true);
      }

      window.addEventListener("keydown", onKey, true);
      window.addEventListener("resize", onResize);
      document.body.append(ov);
      if (img.complete) render(); else img.onload = () => render();
    });
  };

  /** Pin comments of all attachments, numbered like the burned-in markers. */
  OMP.pinNotes = function pinNotes(images) {
    const blocks = [];
    images.forEach((img, n) => {
      if (!img.pins?.length) return;
      const head = images.length > 1 ? `Pins on image ${n + 1}:` : "Pins on the image:";
      const lines = img.pins.map((p, i) =>
        `${i + 1}. (x ${Math.round(p.x * 100)}%, y ${Math.round(p.y * 100)}%) ${(p.text || "").trim() || "(no comment)"}`);
      blocks.push([head, ...lines].join("\n"));
    });
    return blocks.join("\n\n");
  };
})(window.OMP);
