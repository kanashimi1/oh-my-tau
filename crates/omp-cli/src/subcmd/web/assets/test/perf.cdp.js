#!/usr/bin/env node
"use strict";
// Synthetic, no-provider benchmark for the real web UI in a CDP browser.
const fs = require("fs");
const WebSocket = require("ws");
const BASE = process.env.OMP_WEB_URL || "http://127.0.0.1:7467";
const PORT = Number(process.env.OMP_CDP_PORT || 9223);
const N = Number(process.env.OMP_PERF_DELTAS || 600);
const MESSAGES = Number(process.env.OMP_PERF_MESSAGES || 700);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const getJson = (url) => new Promise((resolve, reject) => {
  fetch(url).then((r) => r.json()).then(resolve, reject);
});
let seq = 0, pending = new Map();
let ws;
async function send(method, params = {}, sessionId) {
  const id = ++seq;
  ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}
async function evalPage(sessionId, expression) {
  const r = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true }, sessionId);
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result.value;
}
async function metrics(sessionId) {
  const { metrics } = await send("Performance.getMetrics", {}, sessionId);
  return Object.fromEntries(metrics.map((m) => [m.name, m.value]));
}
function deltaMetric(a, b, key) { return (b[key] || 0) - (a[key] || 0); }
(async () => {
  const version = await getJson(`http://127.0.0.1:${PORT}/json/version`);
  ws = new WebSocket(version.webSocketDebuggerUrl, { perMessageDeflate: false });
  await new Promise((resolve, reject) => { ws.once("open", resolve); ws.once("error", reject); });
  ws.on("message", (raw) => {
    const m = JSON.parse(raw);
    if (!m.id || !pending.has(m.id)) return;
    const p = pending.get(m.id); pending.delete(m.id);
    m.error ? p.reject(new Error(m.error.message)) : p.resolve(m.result);
  });
  const { targetId } = await send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
  await send("Runtime.enable", {}, sessionId);
  await send("Performance.enable", {}, sessionId);
  await send("Page.navigate", { url: BASE }, sessionId);
  for (let i = 0; i < 100 && !(await evalPage(sessionId, "!!window.OMP?.Transcript")); ++i) await sleep(100);
  const setup = await evalPage(sessionId, `(() => {
    window.__longTasks = [];
    new PerformanceObserver((list) => { for (const e of list.getEntries()) __longTasks.push(e.duration); }).observe({type:'longtask', buffered:true});
    const host = document.createElement('div'); const scroller = document.createElement('div');
    scroller.style.cssText = 'position:fixed;left:-20000px;width:900px;height:700px;overflow:auto';
    scroller.append(host); document.body.append(scroller);
    window.__bench = { t: new OMP.Transcript(host, scroller), host, scroller, parseCount: 0, expands: 0, invalid: 0 };
    const queue = __bench.t.decoder.queue; __bench.t.decoder.queue = function(f) { const r=queue.call(this,f); __bench.expands++; if (!r) __bench.invalid++; return r; };
    const parse = JSON.parse; JSON.parse = function(...args) { __bench.parseCount++; return parse.apply(this, args); };
    return !!__bench.t;
  })()`);
  if (!setup) throw new Error("OMP.Transcript unavailable");
  const m0 = await metrics(sessionId);
  const stream = await evalPage(sessionId, `(async () => {
    const t = __bench.t, id = 'perf-stream';
    t.onEvent({type:'message_start', messageId:id, message:{role:'assistant',content:[{type:'text',text:''}]}});
    let message = {role:'assistant',content:[{type:'text',text:''}]};
    t.decoder.seed(message);
    let previous = JSON.stringify(message);
    const t0 = performance.now();
    for (let i = 0; i < ${N}; ++i) {
      message.content[0].text += (i % 12 === 0 ? '\\n\\n' : ' token' + i);
      const next = JSON.stringify(message);
      let p = 0; while (p < previous.length && p < next.length && previous[p] === next[p]) p++;
      let q = 0; while (q < previous.length - p && q < next.length - p && previous[previous.length - 1 - q] === next[next.length - 1 - q]) q++;
      t.onEvent({type:'message_update',messageId:id,messageDelta:{n:previous.length,p,q,s:next.slice(p,next.length-q)}});
      previous = next;
      if (i % 30 === 29) await new Promise(requestAnimationFrame);
    }
    t.onEvent({type:'message_end',messageId:id,message});
    await new Promise(requestAnimationFrame);
    return { elapsedMs:performance.now()-t0, parseCount:__bench.parseCount, expands:__bench.expands, invalid:__bench.invalid, chars:message.content[0].text.length };
  })()`);
  const m1 = await metrics(sessionId);
  const history = await evalPage(sessionId, `(() => {
    const messages = [];
    for (let i=0;i<${MESSAGES};++i) messages.push({role:i%2?'assistant':'user',content:[{type:'text',text:'history '+i+' '+String(i).repeat(50)}],timestamp:i+1});
    const t0=performance.now(); __bench.t.loadLatest(messages);
    let guard=0; while (__bench.t.pendingOlder.length && guard++<20) __bench.t.prependOlder();
    return {elapsedMs:performance.now()-t0,messages:${MESSAGES},topLevelNodes:__bench.host.childElementCount,pendingOlder:__bench.t.pendingOlder.length,pendingNewer:__bench.t.pendingNewer.length};
  })()`);
  await send('HeapProfiler.collectGarbage', {}, sessionId);
  const m2 = await metrics(sessionId);
  const result = {
    deltas:N, stream, history,
    taskDurationMs:deltaMetric(m0,m2,'TaskDuration')*1000,
    scriptDurationMs:deltaMetric(m0,m2,'ScriptDuration')*1000,
    layoutCount:deltaMetric(m0,m2,'LayoutCount'), recalcStyleCount:deltaMetric(m0,m2,'RecalcStyleCount'),
    domNodes:m2.Nodes, jsHeapUsedBytes:m2.JSHeapUsedSize,
    longTasks:await evalPage(sessionId,"__longTasks"),
  };
  console.log(JSON.stringify(result,null,2));
  if (process.env.OMP_PERF_OUT) fs.writeFileSync(process.env.OMP_PERF_OUT,JSON.stringify(result,null,2));
  await send("Target.closeTarget",{targetId});
  ws.close();
})().catch((e) => { console.error(e); if (ws) ws.close(); process.exit(1); });
