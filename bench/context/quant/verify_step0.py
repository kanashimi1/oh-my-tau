#!/usr/bin/env python3
"""Verify whether tjsel is the v7 policy model and compare a few real v7 states."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import onnx
from safetensors import safe_open
from transformers import AutoTokenizer

from common import DEFAULT_TJSEL, add_paths, collate, encode_rows, onnx_predict, read_rows, softmax, torch_predict


def main() -> None:
    ap = argparse.ArgumentParser()
    add_paths(ap)
    ap.add_argument("--onnx", type=Path, default=DEFAULT_TJSEL)
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    graph = onnx.load(args.onnx, load_external_data=False).graph
    initializers = {t.name: list(t.dims) for t in graph.initializer}
    with safe_open(args.ckpt / "model.safetensors", framework="np") as f:
        ckpt_shapes = {k: f.get_slice(k).get_shape() for k in f.keys()}
    comparisons = {}
    for suffix in ("type_emb.weight", "act_head.0.weight", "scorer.1.weight", "act_head.2.weight"):
        matches = [shape for name, shape in initializers.items() if name.endswith(suffix)]
        comparisons[suffix] = {"ckpt": ckpt_shapes[suffix], "onnx": matches,
                               "shape_match": matches == [ckpt_shapes[suffix]]}
    tok = AutoTokenizer.from_pretrained(args.ckpt / "tokenizer")
    cfg = json.loads((args.ckpt / "rl_agent_config.json").read_text())
    rows = read_rows(args.data / "val.jsonl")[:args.limit]
    encoded = encode_rows(tok, cfg, rows)
    onnx_logits, _act, _times = onnx_predict(args.onnx, args.ckpt, encoded, batch_size=1)
    del tok
    gc.collect()
    torch_logits, _act = torch_predict(args.ckpt, encoded, batch_size=1, device_name="cpu")
    result = {
        "ckpt": str(args.ckpt), "onnx": str(args.onnx), "weight_comparisons": comparisons,
        "same_model": all(x["shape_match"] for x in comparisons.values()),
        "probes": [{"episode_id": row["episode_id"], "expected": row["expected"]["action"],
                    "v7_torch_logits": t.tolist(), "tjsel_onnx_logits": o.tolist(),
                    "v7_torch_probs": softmax(t[None])[0].tolist(),
                    "tjsel_onnx_probs": softmax(o[None])[0].tolist()}
                   for row, t, o in zip(rows, torch_logits, onnx_logits)],
        "max_abs_logit_diff": float(np.abs(torch_logits - onnx_logits).max()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
