#!/usr/bin/env python3
"""Collect static INT8 activation ranges on real v7 training states without rewriting weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

from onnxruntime.quantization import CalibrationMethod
from onnxruntime.quantization.calibrate import MinMaxCalibrater

from common import DEFAULT_CKPT, DEFAULT_DATA
from quantize import StateCalibrationReader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--ranges", type=Path, required=True)
    ap.add_argument("--augmented", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--calibration-states", type=int, default=128)
    ap.add_argument("--seed", type=int, default=41873)
    args = ap.parse_args()
    reader = StateCalibrationReader(args.ckpt, args.data, args.calibration_states, args.seed)
    calibrator = MinMaxCalibrater(args.input, op_types_to_calibrate=["MatMul", "Gemm"],
                                  augmented_model_path=args.augmented, use_external_data_format=True,
                                  moving_average=True)
    calibrator.augment_graph()
    calibrator.execution_providers = ["CPUExecutionProvider"]
    # The factory uses an unbounded default thread pool. Match production's two threads and
    # disable duplicate prepacked/arena allocations during this memory-constrained experiment.
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    opts.add_session_config_entry("session.disable_prepacking", "1")
    opts.add_session_config_entry("session.use_env_allocators", "1")
    calibrator.infer_session = ort.InferenceSession(str(args.augmented), sess_options=opts,
                                                    providers=["CPUExecutionProvider"])
    print(f"CALIBRATE states={len(reader.encoded)}", flush=True)
    calibrator.collect_data(reader)
    ranges = calibrator.compute_data()
    with args.ranges.open("wb") as f:
        pickle.dump(ranges, f, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {"states": len(reader.encoded), "seed": args.seed, "method": "MinMax",
                "episode_ids": reader.episode_ids, "tensors": len(ranges)}
    args.ranges.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"CALIBRATE done tensors={len(ranges)}", flush=True)


if __name__ == "__main__":
    main()
