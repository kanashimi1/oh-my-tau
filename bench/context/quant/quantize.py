#!/usr/bin/env python3
"""Per-channel dynamic or calibrated static INT8 quantization of a pruned v7 ONNX."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re

import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_dynamic, quantize_static)
from transformers import AutoTokenizer

from common import DEFAULT_CKPT, DEFAULT_DATA, collate, encode_rows, read_rows


class StateCalibrationReader(CalibrationDataReader):
    def __init__(self, ckpt: Path, data: Path, count: int, seed: int):
        cfg = json.loads((ckpt / "rl_agent_config.json").read_text())
        tok = AutoTokenizer.from_pretrained(ckpt / "tokenizer")
        rows = read_rows(data / "train.jsonl")
        # Preserve class balance in the calibration set, then cover each class in length order.
        by_label = defaultdict(list)
        for row in rows:
            by_label[row["expected"]["action"]].append(row)
        rng = random.Random(seed)
        selected = []
        per_class = max(1, count // len(by_label))
        for label in sorted(by_label):
            group = by_label[label]
            rng.shuffle(group)
            selected.extend(group[:per_class])
        rng.shuffle(selected)
        selected = selected[:count]
        self.encoded = encode_rows(tok, cfg, selected)
        self.pad_id = tok.pad_token_id
        self.order = sorted(range(len(self.encoded)), key=lambda i: len(self.encoded[i]["ids"]))
        self.position = 0
        self.episode_ids = [row["episode_id"] for row in selected]

    def get_next(self):
        if self.position >= len(self.order):
            return None
        item = self.encoded[self.order[self.position]]
        self.position += 1
        return collate([item], self.pad_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("dynamic", "static"), required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--calibration-states", type=int, default=256)
    ap.add_argument("--seed", type=int, default=41873)
    ap.add_argument("--quant-gather", action="store_true")
    ap.add_argument("--exclude-regex", action="append", default=[],
                    help="exclude a MatMul/Gemm/Gather when regex matches its name or initializer inputs")
    ap.add_argument("--exclude-node", action="append", default=[], help="exact ONNX node name to keep unquantized")
    ap.add_argument("--percentile", type=float, default=0.0, help="static Percentile calibration, e.g. 99.9")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = None
    excluded = list(args.exclude_node)
    if args.exclude_regex:
        model = onnx.load(str(args.input))
        model.graph.ClearField("value_info")  # stale inferred types break ORT quantization shape handling
        for node in model.graph.node:
            description = " ".join([node.name, *node.input])
            if node.op_type in ("MatMul", "Gemm", "Gather") and any(
                    re.search(pattern, description) for pattern in args.exclude_regex):
                if not node.name:
                    raise ValueError(f"cannot exclude unnamed node: {description}")
                excluded.append(node.name)
    op_types = ["MatMul", "Gemm"] + (["Gather"] if args.quant_gather else [])
    common = dict(model_input=model if model is not None else str(args.input), model_output=str(args.output),
                  op_types_to_quantize=op_types, per_channel=True, reduce_range=False, weight_type=QuantType.QInt8,
                  nodes_to_exclude=excluded, use_external_data_format=True)
    print(f"QUANT start mode={args.mode} ops={op_types} excluded={len(excluded)}", flush=True)
    if args.mode == "dynamic":
        quantize_dynamic(**common, extra_options={"MatMulConstBOnly": True})
        calibration = None
    else:
        reader = StateCalibrationReader(args.ckpt, args.data, args.calibration_states, args.seed)
        method = CalibrationMethod.Percentile if args.percentile else CalibrationMethod.MinMax
        extra = {"CalibMovingAverage": True}
        if args.percentile:
            extra["CalibPercentile"] = args.percentile
        quantize_static(**common, calibration_data_reader=reader, quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, calibrate_method=method,
                        calibration_providers=["CPUExecutionProvider"], extra_options=extra)
        calibration = {"states": len(reader.encoded), "seed": args.seed, "method": str(method),
                       "percentile": args.percentile, "episode_ids": reader.episode_ids}
    metadata = {"transformation": f"{args.mode}_int8", "source": str(args.input),
                "per_channel": True, "weight_type": "QInt8", "activation_type": "QUInt8" if args.mode == "static" else None,
                "op_types": op_types, "exclude_regex": args.exclude_regex, "excluded_nodes": excluded,
                "calibration": calibration}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"QUANT done bytes={args.output.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
