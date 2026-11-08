#!/usr/bin/env python3
"""Merge sharded static Min/Max calibration ranges."""
import argparse
from pathlib import Path
import pickle
from onnxruntime.quantization import CalibrationMethod
from onnxruntime.quantization.calibrate import TensorsData

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("parts", type=Path, nargs="+")
args = ap.parse_args()
merged = {}
for part in args.parts:
    with part.open("rb") as f:
        ranges = pickle.load(f)
    overlap = set(merged) & set(ranges.keys())
    if overlap:
        raise ValueError(f"duplicate tensor ranges: {sorted(overlap)[:3]}")
    merged.update(ranges.items())
with args.out.open("wb") as f:
    pickle.dump(TensorsData(CalibrationMethod.MinMax, merged), f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"MERGED tensors={len(merged)} parts={len(args.parts)}")
