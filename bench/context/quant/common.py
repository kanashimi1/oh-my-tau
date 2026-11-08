#!/usr/bin/env python3
"""Shared Laya v7 data encoding, inference, and quality metrics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

ACTIONS = ("noop", "clm_edit", "rlm_offload", "rlm_recall")
MAIN = Path("/home/kaneki/fleet-split/omt")
DEFAULT_CKPT = MAIN / "bench/context/results/laya-recovery-v7/ckpt"
DEFAULT_DATA = MAIN / "bench/context/results/laya-recovery-v7-data"
DEFAULT_TJSEL = MAIN / "crates/omp-judge/onnx/laya_judge_tjsel.onnx"


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def encode_rows(tok, cfg: dict, rows: Iterable[dict]) -> list[dict]:
    from laya.train import encode_item, items_from_rows

    items, skipped = items_from_rows(tok, rows, cfg.get("max_len", 1024), cfg.get("head_max_len", 256))
    if skipped:
        raise ValueError(f"encoding skipped rows: {skipped}")
    return [encode_item(tok, item, cfg.get("max_len", 1024), cfg.get("head_max_len", 256))
            for item in items]


def collate(encoded: list[dict], pad_id: int) -> dict[str, np.ndarray]:
    import torch
    from laya.common import collate_items

    batch = collate_items([encoded], pad_id)
    return {k: batch[k].numpy() for k in
            ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}


def targets(encoded: list[dict]) -> np.ndarray:
    return np.asarray([int(np.argmax(item["target"])) for item in encoded], dtype=np.int64)


def onnx_predict(onnx_path: Path, ckpt: Path, encoded: list[dict], batch_size: int = 8,
                 threads: int = 2, verbose: bool = True) -> tuple[np.ndarray, np.ndarray, list[float]]:
    import time
    import onnxruntime as ort
    from transformers import AutoTokenizer

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    tok = AutoTokenizer.from_pretrained(ckpt / "tokenizer")
    # Runtime also batches by length. Besides matching that behavior, sorting avoids padding
    # every short controller state to the longest state in an arbitrary dataset chunk.
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]["ids"]))
    all_logits: list[np.ndarray | None] = [None] * len(encoded)
    all_act: list[np.ndarray | None] = [None] * len(encoded)
    times = [0.0] * len(encoded)
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        chunk = [encoded[i] for i in indices]
        batch = collate(chunk, tok.pad_token_id)
        before = time.perf_counter()
        logits, act = session.run(["logits", "act_logits"], batch)
        elapsed = time.perf_counter() - before
        for row, original in enumerate(indices):
            all_logits[original] = logits[row]
            all_act[original] = act[row]
            times[original] = elapsed / len(chunk)
        if verbose and (start // batch_size) % 10 == 0:
            print(f"onnx {start + len(chunk)}/{len(encoded)} {elapsed:.3f}s/batch "
                  f"length={batch['input_ids'].shape[1]}", flush=True)
    assert all(x is not None for x in all_logits) and all(x is not None for x in all_act)
    # All policy examples have four options. Keep this evaluator deliberately specific to v7.
    return np.stack(all_logits), np.stack(all_act), times


def torch_predict(ckpt: Path, encoded: list[dict], batch_size: int = 8,
                  device_name: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    import torch
    from laya.common import collate_items
    from laya.train import load_checkpoint

    device = torch.device(device_name)
    model, tok, _cfg = load_checkpoint(str(ckpt))
    model.eval().to(device)
    model.head_checkpointing = False
    all_logits, all_act = [], []
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            chunk = encoded[start:start + batch_size]
            batch = collate_items([chunk], tok.pad_token_id)
            args = [batch[k].to(device) for k in
                    ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
            logits, act = model(*args)
            all_logits.append(logits.float().cpu().numpy())
            all_act.append(act.float().cpu().numpy())
    return np.concatenate(all_logits), np.concatenate(all_act)


def softmax(logits: np.ndarray, mask: np.ndarray | None = None, temperature: float = 1.0) -> np.ndarray:
    z = logits.astype(np.float64) / temperature
    if mask is not None:
        z = np.where(mask, z, -np.inf)
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        selected = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= 0) & (conf <= edges[1])
        if selected.any():
            total += selected.mean() * abs(float(correct[selected].mean()) - float(conf[selected].mean()))
    return total


def quality(candidate_logits: np.ndarray, reference_logits: np.ndarray, labels: np.ndarray,
            mask: np.ndarray | None = None, temperature: float = 1.0) -> dict:
    from sklearn.metrics import f1_score

    cp = softmax(candidate_logits, mask, temperature)
    rp = softmax(reference_logits, mask, temperature)
    ca, ra = cp.argmax(axis=1), rp.argmax(axis=1)
    n = len(labels)
    return {
        "n": n,
        "argmax_agreement": float((ca == ra).mean()),
        "candidate_macro_f1": float(f1_score(labels, ca, average="macro", zero_division=0)),
        "reference_macro_f1": float(f1_score(labels, ra, average="macro", zero_division=0)),
        "macro_f1_delta_pp": float(100 * (f1_score(labels, ca, average="macro", zero_division=0) -
                                           f1_score(labels, ra, average="macro", zero_division=0))),
        "max_prob_diff": float(np.abs(cp - rp).max()),
        "mean_prob_diff": float(np.abs(cp - rp).mean()),
        "candidate_nll": float(-np.log(np.clip(cp[np.arange(n), labels], 1e-12, 1)).mean()),
        "reference_nll": float(-np.log(np.clip(rp[np.arange(n), labels], 1e-12, 1)).mean()),
        "candidate_ece": ece(cp, labels),
        "reference_ece": ece(rp, labels),
        "disagreements": np.flatnonzero(ca != ra).tolist(),
    }


def add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
