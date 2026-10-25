"""L4: replay of a real long session (read-only source) as the starting history.

The prefix of a real omp session (user / assistant text+toolCall / toolResult, no thinking, no images) is
re-serialised into a fresh linear session file, then `--resume`d by omp-rs under the arm's context strategy;
follow-up steps ask recall questions whose ground truth is computed from the prefix itself.
"""
import glob
import hashlib
import json
import os
import random
import re
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "results", "long", "l4cache")
ROOTS = ("/home/kaneki/.omp/agent/sessions", "/home/kaneki/.omp-rs/sessions")
# (glob suffix, target chars of the replayed prefix)
SOURCES = [
    ("it/2026-09-14T19-00-35-799Z_01a0a14a-e757-771c-8669-812369e4e4e6", 200_000),
    ("it/2026-09-15T23-37-05-537Z_01a0a76e-6701-74dc-ae24-bd96854d0df8", 240_000),
    ("it/2026-09-14T11-18-45-311Z_01a09fa4-133f-71a2-8bba-f9f8f7fe623f", 280_000),
    ("it/2026-09-17T16-50-37-479Z_01a0b046-fd27-7486-aaf3-082159d07131", 220_000),
]


def _find(stem):
    for root in ROOTS:
        hits = glob.glob(f"{root}/*/{stem.split('/')[-1]}.jsonl")
        if hits:
            return hits[0]
    raise FileNotFoundError(stem)


def _text(blocks):
    return "\n".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")


def build_prefix(src, target_chars):
    """-> (entries list of message dicts, calls list, results dict id->text)"""
    msgs = []
    for line in open(src, errors="ignore"):
        if '"type":"message"' not in line[:40]:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        msgs.append(d["message"])
    out, calls, chars, last_safe = [], [], 0, None
    seen_calls, pending = set(), set()
    for m in msgs:
        role = m.get("role")
        if role == "user":
            t = _text(m["content"]) if isinstance(m["content"], list) else str(m["content"])
            if not t.strip():
                continue
            m = dict(m, content=[{"type": "text", "text": t[:20000]}])
        elif role == "assistant":
            blocks = [b for b in m["content"] if b.get("type") in ("text", "toolCall")]
            if not blocks:
                continue
            for b in blocks:
                if b["type"] == "toolCall":
                    seen_calls.add(b["id"])
                    pending.add(b["id"])
            m = dict(m, content=blocks)
        elif role == "toolResult":
            if m.get("toolCallId") not in seen_calls:
                continue
            t = _text(m["content"])
            m = dict(m, content=[{"type": "text", "text": t[:30000]}])
            pending.discard(m["toolCallId"])
            for c in reversed(calls):
                if c["id"] == m["toolCallId"]:
                    c["result"] = t
                    break
        else:
            continue
        out.append(m)
        if role == "assistant":
            for b in m["content"]:
                if b["type"] == "toolCall":
                    calls.append({"id": b["id"], "name": b["name"], "args": b.get("arguments") or {}, "result": None})
            chars += sum(len(b.get("text", "")) + len(json.dumps(b.get("arguments", ""))) for b in m["content"])
            if not pending:
                last_safe = len(out)
        elif role == "toolResult":
            chars += len(_text(m["content"]))
            if not pending:
                last_safe = len(out)
        else:
            chars += len(_text(m["content"]))
        if chars >= target_chars and last_safe == len(out):
            break
    out = out[:last_safe] if last_safe else out
    if out and out[-1].get("role") != "assistant":
        last_a = next(x for x in reversed(out) if x.get("role") == "assistant")
        out.append(dict(last_a, content=[{"type": "text", "text": "Ready for the next instruction."}],
                        stopReason="stop"))
    kept = {m["toolCallId"] for m in out if m.get("role") == "toolResult"}
    calls = [c for c in calls if c["id"] in kept and c["result"] is not None]
    return out, calls, chars


def write_session(msgs, path, cwd):
    import datetime
    ts = datetime.datetime(2026, 10, 1, tzinfo=datetime.timezone.utc)
    sid = str(uuid.uuid4())
    with open(path, "w") as f:
        f.write(json.dumps({"type": "session", "version": 3, "id": sid, "timestamp": ts.isoformat().replace("+00:00", "Z"),
                            "cwd": cwd}) + "\n")
        prev = None
        for i, m in enumerate(msgs):
            eid = hashlib.sha1(f"{sid}{i}".encode()).hexdigest()[:8]
            t = (ts + datetime.timedelta(seconds=i)).isoformat().replace("+00:00", "Z")
            f.write(json.dumps({"type": "message", "id": eid, "parentId": prev, "timestamp": t, "message": m},
                               ensure_ascii=False) + "\n")
            prev = eid


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def gen_l4(seed, steps=14):
    stem, target = SOURCES[seed % len(SOURCES)]
    os.makedirs(CACHE, exist_ok=True)
    cp = os.path.join(CACHE, f"s{seed % len(SOURCES)}.json")
    if not os.path.exists(cp):
        msgs, calls, chars = build_prefix(_find(stem), target)
        json.dump({"msgs": msgs, "calls": calls, "chars": chars}, open(cp, "w"), ensure_ascii=False)
    d = json.load(open(cp))
    msgs, calls = d["msgs"], d["calls"]
    sess_path = os.path.join(CACHE, f"s{seed % len(SOURCES)}.session.jsonl")
    write_session(msgs, sess_path, "/tmp")
    rng = random.Random(f"l4-{seed}")
    early = calls[: max(12, len(calls) // 2)]

    def first_line(res):
        ls = res.splitlines()
        if not ls or not ls[0].startswith("["):
            return None
        for l in ls[1:6]:
            t = re.sub(r"^\*?\s*\d+[:|]\s?", "", l).strip()
            if len(t) >= 20:
                return t
        return None

    def base(c):
        return os.path.basename(re.sub(r":[0-9,+\-rawc:]+$", "", c["args"]["path"]))

    reads = [c for c in early if c["name"] == "read" and isinstance(c["args"].get("path"), str)
             and "?" not in c["args"]["path"] and first_line(c["result"] or "")]
    uniq_base = [c for c in reads if sum(1 for x in calls if x["name"] == "read" and isinstance(x["args"].get("path"), str)
                                         and base(x) == base(c)) == 1]
    bashes = [c for c in early if c["name"] == "bash" and isinstance(c["args"].get("command"), str)
              and len(c["args"]["command"]) >= 12
              and any(len(l.strip()) >= 12 for l in (c["result"] or "").splitlines())
              and sum(1 for x in calls if x["name"] == "bash" and x["args"].get("command") == c["args"]["command"]) == 1]
    if len(uniq_base) < 2 or not bashes:  # short prefix: any call of the prefix qualifies
        early = calls
        reads = [c for c in early if c["name"] == "read" and isinstance(c["args"].get("path"), str)
                 and "?" not in c["args"]["path"] and first_line(c["result"] or "")]
        uniq_base = [c for c in reads if sum(1 for x in calls if x["name"] == "read" and isinstance(x["args"].get("path"), str)
                                             and base(x) == base(c)) == 1]
        bashes = [c for c in early if c["name"] == "bash" and isinstance(c["args"].get("command"), str)
                  and len(c["args"]["command"]) >= 12
                  and any(len(l.strip()) >= 12 for l in (c["result"] or "").splitlines())
                  and sum(1 for x in calls if x["name"] == "bash" and x["args"].get("command") == c["args"]["command"]) == 1]
    qa = rng.choice(uniq_base[: max(2, len(uniq_base) // 2)])
    rest = [c for c in uniq_base if c is not qa]
    qc = rng.choice(rest)
    qb = rng.choice(bashes[: max(2, len(bashes) // 2)])
    bl = next(l for l in qb["result"].splitlines() if len(l.strip()) >= 12)
    EN = " Reply in English and keep the literal word ANSWER."
    Qs = {
        "A": ("One of your earlier `read` tool calls returned a file whose first lines began with: \"%s\". Which file path was "
              "that? Reply on one line `ANSWER A: <path>`." % first_line(qa["result"])[:50] + EN, base(qa).lower()),
        "B": ("Your earlier bash call with the command `%s` produced output. What was the first non-empty line of that output? "
              "Reply on one line `ANSWER B: <line>`." % qb["args"]["command"][:100].replace("`", "'") + EN, norm(bl)[:25]),
        "C": ("When you read the file `%s` earlier, what was the first line of its content as returned by the read tool "
              "(ignore the line-number prefix)? Reply on one line `ANSWER C: <line>`." % base(qc) + EN,
              norm(first_line(qc["result"]))[:25]),
    }
    mid = max(3, steps // 2)
    msgs_steps = {}
    for k in range(1, steps):
        msgs_steps[k] = (f"STEP {k}/{steps}: do not use any tools; reply with a one-line status of what this session has done so "
                         f"far [step {k}].")
    msgs_steps[mid] = f"STEP {mid}/{steps}: do not use any tools. " + Qs["A"][0]
    msgs_steps[mid + 2] = f"STEP {mid + 2}/{steps}: do not use any tools. " + Qs["B"][0]
    msgs_steps[steps] = (f"FINAL STEP {steps}/{steps}: do not use any tools. Answer both, one per line: {Qs['C'][0]} "
                         f"{Qs['A'][0]}")
    asked = {mid: ["A"], mid + 2: ["B"], steps: ["C", "A"]}

    def scorer(replies):
        ok, det = 0, []
        for step, ks in asked.items():
            r = norm(replies.get(step, ""))
            for k in ks:
                m = re.findall(rf"answer {k.lower()}:\s*(.+)", r)
                hit = bool(m) and Qs[k][1] in norm(m[-1])
                ok += hit
                det.append(f"{k}{'+' if hit else '-'}")
        n = sum(len(v) for v in asked.values())
        return {"score": ok / n, "pass": ok == n, "detail": " ".join(det), "valid": True}

    return dict(files={}, msgs=msgs_steps, steps=steps, scorer=scorer, sub="relay", timeout=900, resume=sess_path,
                brief="", meta=dict(src=stem, prefix_msgs=len(msgs), prefix_chars=d["chars"], truth={k: v[1] for k, v in Qs.items()}))
