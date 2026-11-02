#!/bin/bash
# Paired fresh-seed closed loop: v4-balanced (v1 states) vs v7 (v2 states, offload_match_pct).
set -e
CTX=/home/kaneki/fleet-split/omt/bench/context
ONNX=/home/kaneki/fleet-split/omt/crates/omp-judge/onnx/laya_judge_tjsel.onnx
cd "$CTX"

export OMP_CONTEXT_LAYA_ONNX=$ONNX
export OMP_CONTEXT_LAYA_CKPT=$CTX/results/laya-recovery-v4-balanced/ckpt
unset OMP_CONTEXT_STATE_V2
python3 longbench.py --family L1 --arms laya --seeds 42-43 --reps 1 --workers 2 --max-spend 21.5 --tag fresh7-v4-
python3 longbench.py --family L2 --arms laya --seeds 42-43 --reps 1 --workers 2 --max-spend 21.5 --tag fresh7-v4-

export OMP_CONTEXT_LAYA_CKPT=$CTX/results/laya-recovery-v7/ckpt
export OMP_CONTEXT_STATE_V2=1
python3 longbench.py --family L1 --arms laya --seeds 42-43 --reps 1 --workers 2 --max-spend 21.5 --tag fresh7-v7-
python3 longbench.py --family L2 --arms laya --seeds 42-43 --reps 1 --workers 2 --max-spend 21.5 --tag fresh7-v7-
echo FRESH7-DONE
