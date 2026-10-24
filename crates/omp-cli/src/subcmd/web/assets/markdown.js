"use strict";
// Minimal, safe Markdown -> HTML (everything is escaped first; only the
// constructs below produce markup) plus a tiny keyword highlighter.
(function (OMP) {
  const esc = OMP.esc;

  const KEYWORDS = new Set((
    "as async await break case catch class const continue crate def default defer del delete do elif else enum " +
    "except export extends false final finally fn for from func function go if impl import in interface is let " +
    "loop match mod move mut new nil none None not null of or package pass pub raise return self Self static " +
    "struct super switch this throw true True False try type typeof undefined unsafe use var void where while " +
    "with yield local then end echo fi done esac"
  ).split(" "));
  const HASH_COMMENT = new Set(["sh", "bash", "zsh", "shell", "py", "python", "rb", "ruby", "toml", "yaml", "yml", "make", "dockerfile", "r", "perl", "conf", "ini"]);

  function highlight(code, lang) {
    lang = (lang || "").toLowerCase();
    if (code.length > 200000 || ["", "text", "txt", "plain", "diff", "patch", "log", "output", "md", "markdown"].includes(lang)) {
      if (lang === "diff" || lang === "patch") return highlightDiff(code);
      return esc(code);
    }
    const hash = HASH_COMMENT.has(lang);
    const re = hash
      ? /(#[^\n]*)|("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'|`(?:[^`\\]|\\.)*`)|(\b\d[\d_.xXa-fA-F]*\b)|([A-Za-z_$][\w$]*)/g
      : /(\/\/[^\n]*|\/\*[\s\S]*?\*\/|<!--[\s\S]*?-->)|("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'|`(?:[^`\\]|\\.)*`)|(\b\d[\d_.xXa-fA-F]*\b)|([A-Za-z_$][\w$]*)/g;
    let out = "", last = 0, m;
    while ((m = re.exec(code))) {
      out += esc(code.slice(last, m.index));
      last = re.lastIndex;
      const [tok, cm, str, num, word] = m;
      if (cm) out += `<span class="tok-cm">${esc(tok)}</span>`;
      else if (str) out += `<span class="tok-str">${esc(tok)}</span>`;
      else if (num) out += `<span class="tok-num">${esc(tok)}</span>`;
      else if (word) {
        if (KEYWORDS.has(word)) out += `<span class="tok-kw">${esc(word)}</span>`;
        else if (code[last] === "(" || code[last] === "!") out += `<span class="tok-fn">${esc(word)}</span>`;
        else if (/^[A-Z][a-z0-9]\w*$/.test(word)) out += `<span class="tok-ty">${esc(word)}</span>`;
        else out += esc(word);
      }
    }
    return out + esc(code.slice(last));
  }

  function highlightDiff(code) {
    return code.split("\n").map((l) => {
      const c = l[0];
      if (c === "+" && !l.startsWith("+++")) return `<span class="add">${esc(l)}</span>`;
      if (c === "-" && !l.startsWith("---")) return `<span class="del">${esc(l)}</span>`;
      if (l.startsWith("@@")) return `<span class="hunk">${esc(l)}</span>`;
      return esc(l) + "\n";
    }).join("").replace(/\n$/, "");
  }
  OMP.highlight = highlight;
  OMP.highlightDiff = highlightDiff;

  function safeUrl(url) {
    const u = url.trim();
    return /^(https?:|mailto:|#|\/)/i.test(u) ? u : null;
  }

  function inline(text) {
    // Code spans first, parked behind placeholders so nothing inside is touched.
    const parked = [];
    const park = (html) => `\u0000${parked.push(html) - 1}\u0000`;
    let s = text.replace(/(`+)([\s\S]*?[^`])\1(?!`)/g, (_, _t, code) => park(`<code>${esc(code)}</code>`));
    s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (m, alt, url) => {
      const u = safeUrl(url);
      return u ? park(`<a href="${esc(u)}" target="_blank" rel="noopener">${esc(alt || u)}</a>`) : m;
    });
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (m, label, url) => {
      const u = safeUrl(url);
      return u ? park(`<a href="${esc(u)}" target="_blank" rel="noopener">${inlineNoLinks(label)}</a>`) : m;
    });
    s = s.replace(/(^|[\s(])(https?:\/\/[^\s<>()]+[^\s<>().,;:!?'"])/g, (_, pre, url) =>
      pre + park(`<a href="${esc(url)}" target="_blank" rel="noopener">${esc(url)}</a>`));
    s = inlineNoLinks(s);
    for (let pass = 0; pass < 3 && s.includes("\u0000"); pass++) s = s.replace(/\u0000(\d+)\u0000/g, (_, i) => parked[Number(i)]);
    return s;
  }

  function inlineNoLinks(s) {
    return esc(s)
      .replace(/\*\*\*([^*\n]+)\*\*\*/g, "<strong><em>$1</em></strong>")
      .replace(/\*\*([^*\n]+?)\*\*/g, "<strong>$1</strong>")
      .replace(/__([^_\n]+?)__/g, "<strong>$1</strong>")
      .replace(/(^|[^*\w])\*([^*\s][^*\n]*?)\*(?!\w)/g, "$1<em>$2</em>")
      .replace(/(^|[^_\w])_([^_\s][^_\n]*?)_(?!\w)/g, "$1<em>$2</em>")
      .replace(/~~([^~\n]+)~~/g, "<del>$1</del>");
  }

  function codeBlock(code, lang) {
    const id = "cb" + Math.random().toString(36).slice(2, 9);
    OMP._codeStore = OMP._codeStore || new Map();
    OMP._codeStore.set(id, code);
    if (OMP._codeStore.size > 3000) OMP._codeStore.delete(OMP._codeStore.keys().next().value);
    return `<div class="codeblock"><div class="code-head"><span>${esc(lang || "")}</span>` +
      `<button class="copy-btn" data-copy="${id}">copy</button></div>` +
      `<pre><code>${highlight(code, lang)}</code></pre></div>`;
  }

  const LIST_RE = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/;

  function blocks(lines) {
    let out = "";
    let i = 0;
    let para = [];
    const flush = () => {
      if (para.length) out += `<p>${inline(para.join("\n")).replace(/\n/g, "<br>")}</p>`;
      para = [];
    };
    while (i < lines.length) {
      const line = lines[i];
      const fence = line.match(/^\s*(`{3,}|~{3,})\s*([\w+#.-]*)/);
      if (fence) {
        flush();
        const marker = fence[1];
        const body = [];
        i++;
        while (i < lines.length && !lines[i].trim().startsWith(marker)) body.push(lines[i++]);
        i++;
        out += codeBlock(body.join("\n"), fence[2]);
        continue;
      }
      if (!line.trim()) { flush(); i++; continue; }
      const h = line.match(/^(#{1,6})\s+(.*?)\s*#*$/);
      if (h) { flush(); out += `<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`; i++; continue; }
      if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flush(); out += "<hr>"; i++; continue; }
      if (/^\s*>/.test(line)) {
        flush();
        const quote = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) quote.push(lines[i++].replace(/^\s*>\s?/, ""));
        out += `<blockquote>${blocks(quote)}</blockquote>`;
        continue;
      }
      if (line.includes("|") && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(lines[i + 1])) {
        flush();
        const cells = (l) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
        const head = cells(line);
        const aligns = cells(lines[i + 1]).map((c) => (c.endsWith(":") ? (c.startsWith(":") ? "center" : "right") : ""));
        i += 2;
        let t = "<table><thead><tr>" + head.map((c, k) => `<th${aligns[k] ? ` style="text-align:${aligns[k]}"` : ""}>${inline(c)}</th>`).join("") + "</tr></thead><tbody>";
        while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
          t += "<tr>" + cells(lines[i]).map((c, k) => `<td${aligns[k] ? ` style="text-align:${aligns[k]}"` : ""}>${inline(c)}</td>`).join("") + "</tr>";
          i++;
        }
        out += t + "</tbody></table>";
        continue;
      }
      const li = line.match(LIST_RE);
      if (li) {
        flush();
        const baseIndent = li[1].length;
        const ordered = /\d/.test(li[2]);
        const start = ordered ? parseInt(li[2], 10) : 1;
        const items = [];
        while (i < lines.length) {
          const m = lines[i].match(LIST_RE);
          if (m && m[1].length <= baseIndent + 1 && m[1].length >= baseIndent - 1 && /\d/.test(m[2]) === ordered) {
            items.push([m[3]]);
            i++;
            continue;
          }
          if (!lines[i].trim()) {
            const next = lines[i + 1];
            if (next !== undefined && (/^\s{2,}\S/.test(next) || (LIST_RE.test(next) && next.match(LIST_RE)[1].length >= baseIndent))) { items[items.length - 1].push(""); i++; continue; }
            break;
          }
          if (/^\s+\S/.test(lines[i]) || LIST_RE.test(lines[i])) {
            if (/^\s+\S/.test(lines[i]) && lines[i].search(/\S/) > baseIndent) { items[items.length - 1].push(lines[i].slice(Math.min(lines[i].search(/\S/), baseIndent + 2))); i++; continue; }
          }
          if (!LIST_RE.test(lines[i]) && !/^\s*(#|>|```|~~~)/.test(lines[i])) { items[items.length - 1].push(lines[i].trim()); i++; continue; }
          break;
        }
        const tag = ordered ? "ol" : "ul";
        out += `<${tag}${ordered && start !== 1 ? ` start="${start}"` : ""}>`;
        for (const it of items) {
          const task = it[0].match(/^\[([ xX])\]\s+(.*)$/);
          if (task) it[0] = (task[1] === " " ? "☐ " : "☑ ") + task[2];
          const inner = it.length === 1 ? inline(it[0]) : blocks(it).replace(/^<p>([\s\S]*?)<\/p>/, "$1");
          out += `<li>${inner}</li>`;
        }
        out += `</${tag}>`;
        continue;
      }
      para.push(line);
      i++;
    }
    flush();
    return out;
  }

  OMP.md = function md(text) {
    return blocks(String(text ?? "").replace(/\r\n?/g, "\n").split("\n"));
  };

  // Delegated copy buttons for code blocks.
  document.addEventListener("click", async (e) => {
    const btn = e.target.closest && e.target.closest("[data-copy]");
    if (!btn) return;
    const code = OMP._codeStore && OMP._codeStore.get(btn.dataset.copy);
    if (code == null) return;
    if (await OMP.copyText(code)) {
      btn.textContent = "copied";
      setTimeout(() => (btn.textContent = "copy"), 1200);
    }
  });
})(window.OMP);
