#!/usr/bin/env python3
"""Convert fp16 QDQ scale initializers to the fp32 type required by ORT."""
import argparse
from pathlib import Path
import onnx
from quantize import fp32_qdq_scales

ap = argparse.ArgumentParser()
ap.add_argument("--input", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()
model = onnx.load(str(args.input))
count = fp32_qdq_scales(model)
# ORT's CPU QuantizeLinear accepts fp32 activations, not fp16. Preserve the
# surrounding fp16 graph and cast only each activation input to Q.
nodes = []
casts = 0
for node in model.graph.node:
    if node.op_type == "QuantizeLinear":
        original = node.input[0]
        cast_output = original + "__qdq_fp32"
        nodes.append(onnx.helper.make_node("Cast", [original], [cast_output],
                                           name=node.name + "__input_fp32", to=onnx.TensorProto.FLOAT))
        node.input[0] = cast_output
        casts += 1
    nodes.append(node)
model.graph.ClearField("node")
model.graph.node.extend(nodes)
print(f"REPAIRED activation_casts={casts}")
# Q/DQ computes in fp32; return to the fp16 type expected by the original MatMul consumers.
nodes = []
dq_casts = 0
for node in model.graph.node:
    nodes.append(node)
    if node.op_type == "DequantizeLinear":
        original = node.output[0]
        dq_output = original + "__dq_fp32"
        node.output[0] = dq_output
        nodes.append(onnx.helper.make_node("Cast", [dq_output], [original],
                                           name=node.name + "__output_fp16", to=onnx.TensorProto.FLOAT16))
        dq_casts += 1
model.graph.ClearField("node")
model.graph.node.extend(nodes)
model.graph.ClearField("value_info")
print(f"REPAIRED dequant_output_casts={dq_casts}")
onnx.save_model(model, str(args.output), save_as_external_data=True,
                all_tensors_to_one_file=True, location=args.output.name + ".data", size_threshold=0)
print(f"REPAIRED scales={count}")
