#!/usr/bin/env python3
"""Evaluate a policy ONNX against the frozen fp32 v7 reference on controller states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from common import add_paths, encode_rows, onnx_predict, quality, read_rows, targets


def bootstrap_noise(probs: np.ndarray, labels: np.ndarray, seed: int = 41873, draws: int = 200) -> dict:
    from common import ece

    rng = np.random.default_rng(seed)
    n = len(labels)
    base_nll = float(-np.log(np.clip(probs[np.arange(n), labels], 1e-12, 1)).mean())
    base_ece = ece(probs, labels)
    nll_delta, ece_delta = [], []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        p, y = probs[idx], labels[idx]
        nll = float(-np.log(np.clip(p[np.arange(n), y], 1e-12, 1)).mean())
        nll_delta.append(abs(nll - base_nll))
        ece_delta.append(abs(ece(p, y) - base_ece))
    return {"draws": draws, "nll_abs_delta_p95": float(np.quantile(nll_delta, .95)),
            "ece_abs_delta_p95": float(np.quantile(ece_delta, .95))}


def main() -> None:
    ap = argparse.ArgumentParser()
    add_paths(ap)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reference", type=Path, help="npz from a prior --save-predictions run")
    ap.add_argument("--predictions", type=Path, help="reuse logits instead of running ONNX inference")
    ap.add_argument("--save-predictions", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    cfg = json.loads((args.ckpt / "rl_agent_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(args.ckpt / "tokenizer")
    rows = read_rows(args.data / f"{args.split}.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    encoded = encode_rows(tok, cfg, rows)
    labels = targets(encoded)
    if args.predictions:
        saved = np.load(args.predictions)
        logits, act_logits, times = saved["logits"], saved["act_logits"], saved["times"].tolist()
    else:
        logits, act_logits, times = onnx_predict(args.onnx, args.ckpt, encoded,
                                                 batch_size=args.batch_size, threads=args.threads)
    temperature = float(cfg.get("temperature", [1.0])[0])
    if args.reference:
        ref = np.load(args.reference)
        reference_logits = ref["logits"]
        reference_labels = ref["labels"]
        if len(reference_labels) != len(labels) and args.limit:
            reference_logits = reference_logits[:len(labels)]
            reference_labels = reference_labels[:len(labels)]
        if not np.array_equal(labels, reference_labels):
            raise ValueError("reference labels differ; use the same split and limit")
        if reference_logits.shape != logits.shape:
            raise ValueError(f"reference shape {reference_logits.shape} != {logits.shape}")
    else:
        reference_logits = logits
    external = Path(str(args.onnx) + ".data")
    model_bytes = args.onnx.stat().st_size + (external.stat().st_size if external.exists() else 0)
    metrics = quality(logits, reference_logits, labels, temperature=temperature)
    from common import softmax
    reference_probs = softmax(reference_logits, temperature=temperature)
    noise = bootstrap_noise(reference_probs, labels)
    metrics.update({
        "split": args.split, "onnx": str(args.onnx), "reference": str(args.reference or args.onnx),
        "temperature": temperature, "onnx_bytes": model_bytes,
        "mean_batch_item_latency_ms": float(np.mean(times) * 1000),
        "calibration_noise": noise,
        "nll_within_noise": bool(abs(metrics["candidate_nll"] - metrics["reference_nll"]) <= noise["nll_abs_delta_p95"]),
        "ece_within_noise": bool(abs(metrics["candidate_ece"] - metrics["reference_ece"]) <= noise["ece_abs_delta_p95"]),
    })
    metrics["passes_policy_gate"] = bool(metrics["argmax_agreement"] >= .995 and
                                            abs(metrics["macro_f1_delta_pp"]) <= .5 and
                                            metrics["nll_within_noise"] and metrics["ece_within_noise"])
    if args.save_predictions:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_predictions, logits=logits, act_logits=act_logits,
                            labels=labels, times=np.asarray(times))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
