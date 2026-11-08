#!/usr/bin/env python3
"""Lossless v7 vocabulary pruning and fp16 ONNX storage transformations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from common import DEFAULT_CKPT


def load(path: Path):
    return onnx.load(str(path), load_external_data=True)


def save(model, path: Path, metadata: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"SAVED {path} bytes={path.stat().st_size}", flush=True)


def largest_embedding(model):
    candidates = [t for t in model.graph.initializer if len(t.dims) == 2 and t.dims[1] >= 768]
    if not candidates:
        raise ValueError("no embedding matrix found")
    return max(candidates, key=lambda t: int(np.prod(t.dims)))


def prune(args) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.ckpt / "tokenizer")
    vocab = tok.get_vocab()
    size = len(tok)
    # A contiguous range is essential: slicing the table preserves IDs without a remap.
    assert set(vocab.values()) == set(range(size)), "tokenizer IDs are not exactly 0..len(tokenizer)-1"
    assert all(0 <= i < size for i in tok.added_tokens_encoder.values())
    model = load(args.input)
    embedding = largest_embedding(model)
    if embedding.dims[0] < size:
        raise ValueError(f"embedding has {embedding.dims[0]} rows, tokenizer needs {size}")
    original_rows = int(embedding.dims[0])
    weights = numpy_helper.to_array(embedding)
    del model.graph.initializer[model.graph.initializer.index(embedding)]
    model.graph.initializer.append(numpy_helper.from_array(weights[:size].copy(), embedding.name))
    # Preserve the original graph's embedding name and output semantics; only unreachable rows vanish.
    tokenizer_json = (args.ckpt / "tokenizer/tokenizer.json").read_bytes()
    save(model, args.output, {
        "transformation": "exact_vocab_prune", "source": str(args.input),
        "embedding": embedding.name, "original_rows": original_rows, "kept_rows": size,
        "removed_rows": original_rows - size,
        "proof": "AutoTokenizer vocabulary and added-token IDs are exactly the contiguous range 0..len(tokenizer)-1",
        "tokenizer_sha256": hashlib.sha256(tokenizer_json).hexdigest(),
    })


def fp16_weights(args) -> None:
    """Store every fp32 initializer as fp16 but cast it back before every consumer.

    This preserves fp32 operator computation. The v7 checkpoint itself stores fp16 weights,
    so the initializer values are bit-exact after this round trip.
    """
    model = load(args.input)
    graph = model.graph
    casts, converted = [], 0
    for initializer in list(graph.initializer):
        if initializer.data_type != TensorProto.FLOAT:
            continue
        name = initializer.name
        array = numpy_helper.to_array(initializer)
        half_name = name + "__storage_fp16"
        cast_name = name + "__compute_fp32"
        replacement = numpy_helper.from_array(array.astype(np.float16), half_name)
        graph.initializer.append(replacement)
        casts.append(helper.make_node("Cast", [half_name], [cast_name], name="cast_weight__" + name,
                                      to=TensorProto.FLOAT))
        for node in graph.node:
            for i, input_name in enumerate(node.input):
                if input_name == name:
                    node.input[i] = cast_name
        graph.initializer.remove(initializer)
        converted += 1
    graph.node.extend(casts)
    graph.ClearField("value_info")
    save(model, args.output, {"transformation": "fp16_initializer_storage_fp32_compute",
                              "source": str(args.input), "converted_initializers": converted})


def fp16_native(args) -> None:
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = load(args.input)
    converted = convert_float_to_float16(model, keep_io_types=True, disable_shape_inference=True,
                                         op_block_list=args.block_op or None)
    save(converted, args.output, {"transformation": "native_fp16", "source": str(args.input),
                                  "keep_io_types": True, "op_block_list": args.block_op})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prune", "fp16-weights", "fp16-native"))
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--block-op", action="append", help="fp16-native ONNX op type to keep in fp32")
    args = ap.parse_args()
    {"prune": prune, "fp16-weights": fp16_weights, "fp16-native": fp16_native}[args.mode](args)


if __name__ == "__main__":
    main()
