#!/usr/bin/env python3
"""Replace verified fp32 graph weights from an architecture-identical v7 checkpoint."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors import safe_open


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--original-ckpt", type=Path, required=True)
    ap.add_argument("--new-ckpt", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    model = onnx.load(str(args.graph), load_external_data=False)
    original = safe_open(str(args.original_ckpt / "model.safetensors"), framework="np")
    new = safe_open(str(args.new_ckpt / "model.safetensors"), framework="np")
    original_keys, new_keys = set(original.keys()), set(new.keys())
    if original_keys != new_keys:
        raise ValueError("checkpoint tensor keys differ")
    by_shape = defaultdict(list)
    for key in original_keys:
        shape = tuple(original.get_slice(key).get_shape())
        by_shape[shape].append((key, False))
        if len(shape) == 2:
            by_shape[shape[::-1]].append((key, True))
    locations = {info.value for initializer in model.graph.initializer for info in initializer.external_data
                 if info.key == "location"}
    if len(locations) != 1:
        raise ValueError(f"expected one external tensor file, got {locations}")
    source_data = args.graph.parent / next(iter(locations))
    output_data = Path(str(args.output) + ".data")
    if output_data.exists():
        raise FileExistsError(output_data)
    shutil.copyfile(source_data, output_data)
    replacements, unmatched = [], []
    mapped_outputs = []
    for index, initializer in enumerate(model.graph.initializer):
        name = initializer.name
        target_key, transpose = (name, False) if name in new_keys else (None, False)
        external = {info.key: info.value for info in initializer.external_data}
        if external:
            dtype = onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type)
            offset = int(external.get("offset", "0"))
            array = np.memmap(source_data, dtype=dtype, mode="r", offset=offset, shape=tuple(initializer.dims))
            output_array = np.memmap(output_data, dtype=dtype, mode="r+", offset=offset, shape=tuple(initializer.dims))
            mapped_outputs.append(output_array)
        else:
            array = numpy_helper.to_array(initializer)
            output_array = None
        if target_key is None and tuple(array.shape) in by_shape:
            wanted = digest(array)
            for key, transposed in by_shape[tuple(array.shape)]:
                candidate = original.get_tensor(key).astype(np.float32)
                if transposed:
                    candidate = candidate.T
                if digest(candidate) == wanted:
                    target_key, transpose = key, transposed
                    break
        if target_key is None:
            unmatched.append(name)
            continue  # shape/scale constants are not checkpoint weights
        replacement = new.get_tensor(target_key).astype(array.dtype)
        if transpose:
            replacement = replacement.T
        if replacement.shape != array.shape:
            raise ValueError(f"shape mismatch for {name} <- {target_key}: {replacement.shape} vs {array.shape}")
        if output_array is None:
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(
                np.ascontiguousarray(replacement), name))
        else:
            output_array[...] = replacement
        replacements.append({"onnx": name, "checkpoint": target_key, "transposed": transpose})
    if len(replacements) < 150:
        raise ValueError(f"only {len(replacements)} checkpoint weights mapped; refusing incomplete replacement")
    for output_array in mapped_outputs:
        output_array.flush()
    onnx.save_model(model, str(args.output), save_as_external_data=False)
    metadata = {"graph": str(args.graph), "original_ckpt": str(args.original_ckpt),
                "new_ckpt": str(args.new_ckpt), "replacements": replacements,
                "non_weight_initializers_left_unchanged": unmatched}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"REPLACED weights={len(replacements)} unchanged={len(unmatched)}", flush=True)


if __name__ == "__main__":
    main()
