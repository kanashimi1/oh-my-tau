#!/usr/bin/env python3
"""Memory-bounded per-channel weight-only QDQ export for an external-data fp32 graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx


def external_info(tensor):
    return {info.key: info.value for info in tensor.external_data}


def set_external(tensor, location: str, offset: int, length: int) -> None:
    tensor.data_location = onnx.TensorProto.EXTERNAL
    tensor.external_data.clear()
    for key, value in (("location", location), ("offset", str(offset)), ("length", str(length))):
        tensor.external_data.add(key=key, value=value)
    tensor.raw_data = b""


def make_external(name, dims, dtype, location, offset, length):
    tensor = onnx.TensorProto()
    tensor.name = name
    tensor.dims.extend(dims)
    tensor.data_type = dtype
    set_external(tensor, location, offset, length)
    return tensor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--quant-gather", action="store_true")
    ap.add_argument("--exclude-node", action="append", default=[])
    args = ap.parse_args()
    model = onnx.load(str(args.input), load_external_data=False)
    initializers = {t.name: t for t in model.graph.initializer}
    targets = {}
    for node in model.graph.node:
        if node.name in args.exclude_node:
            continue
        if node.op_type in ("MatMul", "Gemm") and node.input[1] in initializers:
            targets[node.input[1]] = 1  # ONNX MatMul B is [input, output]
        elif args.quant_gather and node.op_type == "Gather" and "tok_embeddings" in node.input[0]:
            targets[node.input[0]] = 0
    locations = {external_info(t).get("location") for t in model.graph.initializer if external_info(t)}
    if len(locations) != 1:
        raise ValueError(f"expected one external file, got {locations}")
    source_data = args.input.parent / next(iter(locations))
    output_data = Path(str(args.output) + ".data")
    if args.output.exists() or output_data.exists():
        raise FileExistsError(args.output)
    location = output_data.name
    new_initializers, dq_nodes, records = [], [], []
    with source_data.open("rb") as source, output_data.open("wb") as output:
        for initializer in model.graph.initializer:
            external = external_info(initializer)
            if not external:
                continue
            offset, length = int(external.get("offset", "0")), int(external["length"])
            if initializer.name not in targets:
                set_external(initializer, location, output.tell(), length)
                source.seek(offset)
                remaining = length
                while remaining:
                    block = source.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        raise EOFError(initializer.name)
                    output.write(block)
                    remaining -= len(block)
                continue
            axis = targets[initializer.name]
            dtype = onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type)
            weights = np.memmap(source_data, dtype=dtype, mode="r", offset=offset, shape=tuple(initializer.dims))
            moved = weights if axis == 0 else weights.T
            flat = moved.reshape(moved.shape[0], -1)
            scale = (flat.astype(np.float32).__abs__().max(axis=1) / 127.0).astype(np.float32)
            scale = np.maximum(scale, np.float32(1e-12))
            quantized = np.rint(flat / scale[:, None]).clip(-127, 127).astype(np.int8)
            quantized = quantized if axis == 0 else quantized.T
            q_name, s_name, z_name = (initializer.name + suffix for suffix in
                                      ("_quantized", "_scale", "_zero_point"))
            start = output.tell()
            output.write(quantized.tobytes())
            new_initializers.append(make_external(q_name, initializer.dims, onnx.TensorProto.INT8,
                                                  location, start, output.tell() - start))
            start = output.tell()
            output.write(scale.tobytes())
            new_initializers.append(make_external(s_name, scale.shape, onnx.TensorProto.FLOAT,
                                                  location, start, output.tell() - start))
            zero = np.zeros(scale.shape, dtype=np.int8)
            start = output.tell()
            output.write(zero.tobytes())
            new_initializers.append(make_external(z_name, zero.shape, onnx.TensorProto.INT8,
                                                  location, start, output.tell() - start))
            dq_nodes.append(onnx.helper.make_node("DequantizeLinear", [q_name, s_name, z_name],
                                                  [initializer.name], name=initializer.name + "__dequantize",
                                                  axis=axis))
            records.append({"weight": initializer.name, "axis": axis, "shape": list(initializer.dims),
                            "channels": len(scale)})
            initializer.data_type = onnx.TensorProto.INT8  # marker for removal below
        # Remove only original tensors replaced by their quantized representations.
        kept = [t for t in model.graph.initializer if t.name not in targets]
        model.graph.ClearField("initializer")
        model.graph.initializer.extend(kept)
        model.graph.initializer.extend(new_initializers)
    original_nodes = list(model.graph.node)
    model.graph.ClearField("node")
    model.graph.node.extend(dq_nodes)
    model.graph.node.extend(original_nodes)
    model.graph.ClearField("value_info")
    onnx.save_model(model, str(args.output), save_as_external_data=False)
    metadata = {"source": str(args.input), "quantized_weights": records, "weight_type": "QInt8",
                "compute": "fp32 activations and MatMul/Gather, per-channel weight QDQ",
                "excluded_nodes": args.exclude_node}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    total = args.output.stat().st_size + output_data.stat().st_size
    print(f"STREAM_QUANT weights={len(records)} total_bytes={total}", flush=True)


if __name__ == "__main__":
    main()
