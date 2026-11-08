#!/usr/bin/env python3
"""Compare candidate and fp32 judge-sessions JSONL on identical real episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def load(path: Path) -> dict:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return {(r["session_id"], r["turn_index"]): r for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ref, cand = load(args.reference), load(args.candidate)
    if set(ref) != set(cand):
        raise ValueError(f"episode keys differ: missing={list(set(ref)-set(cand))[:3]} extra={list(set(cand)-set(ref))[:3]}")
    keys = sorted(ref)
    result = {"episodes": len(keys)}
    for field in ("success_p", "rework_p"):
        r = np.array([ref[k][field] for k in keys])
        c = np.array([cand[k][field] for k in keys])
        result[field] = {"pearson": float(np.corrcoef(r, c)[0, 1]), "max_abs_diff": float(np.abs(r-c).max()),
                         "mean_abs_diff": float(np.abs(r-c).mean()),
                         "reference_range": [float(r.min()), float(r.max())]}
    options = list(next(iter(ref.values()))["failure_mode_probs"])
    r = np.array([[ref[k]["failure_mode_probs"][o] for o in options] for k in keys])
    c = np.array([[cand[k]["failure_mode_probs"][o] for o in options] for k in keys])
    ra, ca = r.argmax(1), c.argmax(1)
    result["failure_mode"] = {
        "argmax_agreement": float((ra == ca).mean()), "max_abs_prob_diff": float(np.abs(r-c).max()),
        "candidate_macro_f1_vs_reference": float(f1_score(ra, ca, average="macro", zero_division=0)),
        "disagreements": int((ra != ca).sum()),
    }
    result["passes_real_gate"] = bool(result["failure_mode"]["argmax_agreement"] >= .995 and
                                         result["failure_mode"]["candidate_macro_f1_vs_reference"] >= .995)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
