#!/usr/bin/env python3
"""Memory-bounded static calibration: augment in one process, stream ranges in another."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import onnx
from onnx import TensorProto
from onnxruntime.quantization import CalibrationMethod
from onnxruntime.quantization.calibrate import TensorsData

from common import DEFAULT_CKPT, DEFAULT_DATA
from quantize import StateCalibrationReader


def augment(args) -> None:
    model = onnx.load(str(args.input), load_external_data=False)
    value_infos = {vi.name: vi for vi in model.graph.value_info}
    value_infos.update({vi.name: vi for vi in model.graph.output})
    value_infos.update({vi.name: vi for vi in model.graph.input})
    initializers = {t.name for t in model.graph.initializer}
    selected = set()
    for node in model.graph.node:
        if node.op_type not in ("MatMul", "Gemm"):
            continue
        for name in [*node.input, *node.output]:
            vi = value_infos.get(name)
            if name not in initializers and (vi is None or (vi.type.HasField("tensor_type") and
                    vi.type.tensor_type.elem_type == TensorProto.FLOAT)):
                selected.add(name)
    selected = sorted(selected)
    if args.tensor_count:
        selected = selected[args.tensor_offset:args.tensor_offset + args.tensor_count]
    elif args.tensor_offset:
        selected = selected[args.tensor_offset:]
    for name in selected:
        if name in value_infos:
            model.graph.output.add().CopyFrom(value_infos[name])
        else:
            model.graph.output.add().CopyFrom(onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    onnx.save_model(model, str(args.augmented), save_as_external_data=False)
    args.ranges.with_suffix(".json").write_text(json.dumps({"tensors": sorted(selected)}, indent=2) + "\n")
    print(f"AUGMENT tensors={len(selected)}", flush=True)


def collect(args) -> None:
    import onnxruntime as ort
    names = json.loads(args.ranges.with_suffix(".json").read_text())["tensors"]
    reader = StateCalibrationReader(args.ckpt, args.data, args.calibration_states, args.seed)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.augmented), sess_options=opts, providers=["CPUExecutionProvider"])
    ranges = {name: [np.inf, -np.inf] for name in names}
    for index in range(len(reader.encoded)):
        inputs = reader.get_next()
        assert inputs is not None
        values = session.run(names, inputs)
        for name, value in zip(names, values):
            lowest, highest = np.float32(np.nanmin(value)), np.float32(np.nanmax(value))
            ranges[name][0] = min(ranges[name][0], lowest)
            ranges[name][1] = max(ranges[name][1], highest)
        if (index + 1) % 8 == 0:
            print(f"CALIBRATE {index+1}/{len(reader.encoded)}", flush=True)
    tensor_data = TensorsData(CalibrationMethod.MinMax, {k: tuple(v) for k, v in ranges.items()})
    with args.ranges.open("wb") as f:
        pickle.dump(tensor_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    args.ranges.with_suffix(".json").write_text(json.dumps({
        "states": len(reader.encoded), "seed": args.seed, "method": "MinMax",
        "tensors": len(ranges), "episode_ids": reader.episode_ids}, indent=2) + "\n")
    print(f"CALIBRATE done tensors={len(ranges)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("augment", "collect"))
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--ranges", type=Path, required=True)
    ap.add_argument("--augmented", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--calibration-states", type=int, default=128)
    ap.add_argument("--seed", type=int, default=41873)
    ap.add_argument("--tensor-offset", type=int, default=0)
    ap.add_argument("--tensor-count", type=int, default=0)
    args = ap.parse_args()
    (augment if args.mode == "augment" else collect)(args)


if __name__ == "__main__":
    main()
