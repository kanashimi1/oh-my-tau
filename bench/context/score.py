"""Task scoring (port of the clm-blackbox scorer; t1/t2/t3 rules unchanged)."""
import re


def score_t1(reply, ans):
    """All 8 `ANCHOR-<n> code=<c> value=<v>` lines present, verbatim, correct values."""
    got = {}
    for m in re.finditer(r"ANCHOR-(\d+)\s+code=(C\d+)\s+value=([\d.]+)", reply):
        got[f"ANCHOR-{m.group(1)}"] = (m.group(2), float(m.group(3)))
    wrong = [k for k, (c, v) in got.items()
             if k not in ans or ans[k][0] != c or abs(float(ans[k][1]) - v) > 0.05]
    return {"verdict": len(got) == 8 and not wrong, "detail": f"anchors={len(got)}/8 wrong={len(wrong)}"}


def score_t2(reply, ans):
    mt = re.search(r"top3=([a-z]+):(\d+),([a-z]+):(\d+),([a-z]+):(\d+)", reply)
    ms = re.search(r"total=(\d+)", reply)
    if not mt or not ms:
        return {"verdict": False, "detail": "format miss"}
    got = [(mt.group(1), int(mt.group(2))), (mt.group(3), int(mt.group(4))),
           (mt.group(5), int(mt.group(6)))]
    exp = [tuple(x) for x in ans["top3"]]
    total_ok = int(ms.group(1)) == ans["total"]
    return {"verdict": got == exp and total_ok, "detail": f"top3_ok={got == exp} total_ok={total_ok}"}


def score_t3(reply, ans, step_replies):
    """PIN in the final answer; no step reply flips to ABORT (clm-blackbox rule: any 'abort')."""
    pin_ok = ans["pin"] in reply
    aborted = any("abort" in (r or "").lower() for r in step_replies)
    return {"verdict": pin_ok and not aborted, "detail": f"pin_ok={pin_ok} aborted={aborted}"}


def score(task_id, final_reply, ans, step_replies):
    if task_id.startswith("t1"):
        return score_t1(final_reply, ans)
    if task_id.startswith("t2"):
        return score_t2(final_reply, ans)
    return score_t3(final_reply, ans, step_replies)
