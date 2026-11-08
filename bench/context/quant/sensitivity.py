#!/usr/bin/env python3
"""GPU per-layer INT8 weight fake-quant sensitivity on v7 validation states."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from laya.common import collate_items
from laya.train import load_checkpoint
from transformers import AutoTokenizer

from common import DEFAULT_CKPT, DEFAULT_DATA, encode_rows, quality, read_rows, targets


def fake_quant_per_channel(weight: torch.Tensor) -> torch.Tensor:
    flat = weight.detach().float().reshape(weight.shape[0], -1)
    scale = flat.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q = torch.round(flat / scale[:, None]).clamp(-127, 127)
    return (q * scale[:, None]).reshape_as(weight).to(weight.dtype)


def groups(model) -> dict[str, list[torch.nn.Parameter]]:
    result = defaultdict(list)
    for name, parameter in model.named_parameters():
        if "weight" not in name:
            continue
        if name == "encoder.embeddings.word_embeddings.weight":
            group = "embedding"
        elif name.startswith("encoder.layers."):
            group = ".".join(name.split(".")[:3])
        elif name.startswith("head.layers."):
            group = ".".join(name.split(".")[:3])
        elif name.startswith("scorer."):
            group = "scorer"
        elif name.startswith("act_head."):
            group = "act_head"
        elif name == "type_emb.weight":
            group = "type_emb"
        else:
            group = name
        result[group].append(parameter)
    return dict(result)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    model, tok, cfg = load_checkpoint(str(args.ckpt))
    model.eval().to(device)
    rows = read_rows(args.data / "val.jsonl")[:args.limit]
    encoded = encode_rows(tok, cfg, rows)
    labels = targets(encoded)
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]["ids"]))
    batches = []
    for start in range(0, len(order), args.batch_size):
        indices = order[start:start + args.batch_size]
        batches.append((indices, collate_items([[encoded[i] for i in indices]], tok.pad_token_id)))

    def predict():
        output = [None] * len(encoded)
        with torch.inference_mode():
            for indices, batch in batches:
                inputs = [batch[k].to(device) for k in
                          ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
                logits, _act = model(*inputs)
                for row, original in enumerate(indices):
                    output[original] = logits[row].float().cpu().numpy()
        return np.stack(output)

    print("SENSITIVITY teacher", flush=True)
    reference = predict()
    temperature = float(cfg.get("temperature", [1.0])[0])
    results = []
    for index, (name, parameters) in enumerate(groups(model).items()):
        originals = [p.detach().clone() for p in parameters]
        with torch.no_grad():
            for parameter in parameters:
                parameter.copy_(fake_quant_per_channel(parameter))
        logits = predict()
        with torch.no_grad():
            for parameter, original in zip(parameters, originals):
                parameter.copy_(original)
        metrics = quality(logits, reference, labels, temperature=temperature)
        results.append({"group": name, "parameters": len(parameters), **metrics})
        print(f"SENSITIVITY {index+1} {name} agreement={metrics['argmax_agreement']:.5f} "
              f"max_dp={metrics['max_prob_diff']:.6f}", flush=True)
    results.sort(key=lambda r: (r["argmax_agreement"], -r["max_prob_diff"]))
    payload = {"ckpt": str(args.ckpt), "split": "val", "n": len(encoded),
               "method": "symmetric per-output-channel INT8 quantize-dequantize, fp32 activations",
               "groups": results}
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps([{k: r[k] for k in ("group", "argmax_agreement", "max_prob_diff")}
                      for r in results], indent=2))


if __name__ == "__main__":
    main()
