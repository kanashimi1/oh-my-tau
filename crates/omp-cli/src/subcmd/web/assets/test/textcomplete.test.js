"use strict";
// Unit tests for the composer inline text completion (OMP.textComplete.match):
// newest-first prefix matching, dedupe, newline-truncated tails, command-mode
// exclusion and the minimum draft length. The UI path (ghost render + Tab
// accept) is covered by the CDP real-key run.
// Run: node test/textcomplete.test.js

const assert = require("assert");
const fs = require("fs");
const path = require("path");

// composer.js is a classic browser script: evaluate it with a stub OMP.
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
global.window = { OMP: { el: () => ({}), esc, $: () => null } };
new Function("window", fs.readFileSync(path.resolve(__dirname, "../composer.js"), "utf8"))(global.window);

const { match } = global.window.OMP.textComplete;

// ok <name> lines mirror the other suites' reporting style.
let passed = 0;
function ok(name, fn) { fn(); passed++; console.log("ok  " + name); }

ok("prefix tail from the newest matching candidate", () => {
  assert.strictEqual(match("сдел", ["не то", "сделай ещё автодополнение"]), "ай ещё автодополнение");
});
ok("newest first when several match", () => {
  assert.strictEqual(match("ab", ["ab2", "ab1"]), "2");
});
ok("session prompts precede global history", () => {
  assert.strictEqual(match("aa", ["aa-session", "aa-history"]), "-session");
});
ok("case-insensitive draft, candidate casing kept", () => {
  assert.strictEqual(match("СДЕЛ", ["сделай"]), "ай");
});
ok("identical draft is not completed", () => {
  assert.strictEqual(match("ab", ["ab"]), "");
});
ok("shorter or equal candidates are skipped", () => {
  assert.strictEqual(match("abc", ["ab", "abc", "abcd"]), "d");
});
ok("tail stops at the first newline", () => {
  assert.strictEqual(match("ab", ["ab c\nd"]), " c");
});
ok("tail starting with a newline is not offered", () => {
  assert.strictEqual(match("ab", ["ab\ncd"]), "");
});
ok("whitespace-only tails are not offered", () => {
  assert.strictEqual(match("ab", ["ab   "]), "");
});
ok("slash drafts stay with the command autocomplete", () => {
  assert.strictEqual(match("/com", ["/compact all"]), "");
});
ok("drafts shorter than two characters do not complete", () => {
  assert.strictEqual(match("a", ["abc"]), "");
});
ok("no match / no candidates yield an empty tail", () => {
  assert.strictEqual(match("zzz", ["abc"]), "");
  assert.strictEqual(match("ab", []), "");
  assert.strictEqual(match("ab", null), "");
});
ok("non-string candidates are ignored", () => {
  assert.strictEqual(match("ab", [null, 42, undefined, "abc"]), "c");
});
ok("duplicate completions return once", () => {
  assert.strictEqual(match("ab", ["abc", "abc", "abc d"]), "c");
});

console.log("textcomplete.test.js: all checks passed");
