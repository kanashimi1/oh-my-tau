#!/usr/bin/env python3
"""Export the actual v7 context-policy checkpoint to dynamic-shape fp32 ONNX."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from common import DEFAULT_CKPT, DEFAULT_DATA, collate, read_rows
from laya.train import encode_item, items_from_rows, load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model, tok, cfg = load_checkpoint(str(args.ckpt))
    model.eval()
    model.head_checkpointing = False
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rows = read_rows(args.data / "val.jsonl")
    items, skipped = items_from_rows(tok, rows[:1], cfg.get("max_len", 1024), cfg.get("head_max_len", 256))
    assert not skipped and len(items) == 1
    encoded = encode_item(tok, items[0], cfg.get("max_len", 1024), cfg.get("head_max_len", 256))
    arrays = collate([encoded], tok.pad_token_id)
    names = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    example = tuple(torch.from_numpy(arrays[k]) for k in names)
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "marker_pos": {0: "batch", 1: "options"},
        "marker_mask": {0: "batch", 1: "options"},
        "qtype": {0: "batch"},
        "logits": {0: "batch", 1: "options"},
        "act_logits": {0: "batch"},
    }
    print(f"EXPORT start {args.ckpt} -> {args.out}", flush=True)
    torch.onnx.export(model, example, str(args.out), input_names=list(names),
                      output_names=["logits", "act_logits"], dynamic_axes=dynamic_axes,
                      opset_version=args.opset, do_constant_folding=True, dynamo=False)
    metadata = {"source_ckpt": str(args.ckpt), "opset": args.opset, "compute": "fp32",
                "source_config": cfg}
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"EXPORT done bytes={args.out.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
