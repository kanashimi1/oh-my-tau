#!/usr/bin/env python3
"""One-epoch per-channel weight-fake-quant QAT with fp32 v7 logit distillation."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from laya.common import collate_items
from laya.train import load_checkpoint, save_checkpoint, soft_ce_loss
from torch.nn.utils import parametrize

from common import DEFAULT_CKPT, DEFAULT_DATA, encode_rows, quality, read_rows, targets


class PerChannelWeightFakeQuant(torch.nn.Module):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        flat = weight.reshape(weight.shape[0], -1)
        scale = flat.detach().abs().amax(dim=1).clamp_min(1e-12).float() / 127.0
        scale = scale.to(weight.dtype)
        quantized = torch.round(flat / scale[:, None]).clamp(-127, 127) * scale[:, None]
        quantized = quantized.reshape_as(weight)
        return weight + (quantized - weight).detach()


def fake_quant_encoder(model) -> list:
    handles = []
    for name, module in model.encoder.named_modules():
        if isinstance(module, torch.nn.Linear) or module is model.encoder.embeddings.tok_embeddings:
            parametrize.register_parametrization(module, "weight", PerChannelWeightFakeQuant())
            handles.append(module)
    return handles


def batch_for(model, tok, encoded, indices, device):
    batch = collate_items([[encoded[i] for i in indices]], tok.pad_token_id)
    args = [batch[k].to(device) for k in
            ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
    return batch, args


@torch.inference_mode()
def predict(model, tok, encoded, device, batch_size=16):
    model.eval()
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]["ids"]))
    output = [None] * len(encoded)
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        _batch, args = batch_for(model, tok, encoded, indices, device)
        logits, _act = model(*args)
        for row, original in enumerate(indices):
            output[original] = logits[row].float().cpu().numpy()
    return np.stack(output)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--teacher-cache", type=Path, required=True)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--encoder-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-5)
    ap.add_argument("--ce-weight", type=float, default=0.2, help="gold soft-CE weight; 0 is pure distillation")
    ap.add_argument("--seed", type=int, default=41873)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Encode once with a temporary tokenizer; model loads below do not duplicate dataset state.
    from transformers import AutoTokenizer
    cfg = json.loads((args.ckpt / "rl_agent_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(args.ckpt / "tokenizer")
    train = encode_rows(tok, cfg, read_rows(args.data / "train.jsonl"))
    val_rows = read_rows(args.data / "val.jsonl")
    val = encode_rows(tok, cfg, val_rows)
    train_labels, val_labels = targets(train), targets(val)
    if args.teacher_cache.exists():
        cached = np.load(args.teacher_cache)
        teacher_train, teacher_val = cached["train_logits"], cached["val_logits"]
        if not np.array_equal(cached["train_labels"], train_labels) or not np.array_equal(cached["val_labels"], val_labels):
            raise ValueError("teacher cache labels differ")
        print("QAT teacher cache reused", flush=True)
    else:
        teacher, tok, cfg = load_checkpoint(str(args.ckpt))
        teacher.eval().to(device)
        print("QAT fp32 teacher train inference", flush=True)
        teacher_train = predict(teacher, tok, train, device)
        print("QAT fp32 teacher val inference", flush=True)
        teacher_val = predict(teacher, tok, val, device)
        args.teacher_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.teacher_cache, train_logits=teacher_train, val_logits=teacher_val,
                            train_labels=train_labels, val_labels=val_labels)
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
    student, tok, cfg = load_checkpoint(str(args.ckpt))
    quantized_modules = fake_quant_encoder(student)
    student.to(device)
    student.train()
    student.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.head_checkpointing = True
    for parameter in student.encoder.parameters():
        parameter.requires_grad_(True)
    encoder_params = [p for n, p in student.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head_params = [p for n, p in student.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.encoder_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=0.01)
    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)
    steps_per_epoch = math.ceil(len(order) / args.micro_batch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=math.ceil(steps_per_epoch / args.grad_accum), eta_min=1e-6)
    teacher_train_t = torch.from_numpy(teacher_train)
    optimizer.zero_grad(set_to_none=True)
    total_loss, updates = 0.0, 0
    for step, start in enumerate(range(0, len(order), args.micro_batch)):
        indices = order[start:start + args.micro_batch]
        batch, inputs = batch_for(student, tok, train, indices, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _act = student(*inputs)
        logits = logits.float()
        mask = batch["marker_mask"].to(device)
        target = batch["target"].to(device)
        teacher_logits = teacher_train_t[indices].to(device)
        teacher_probs = torch.softmax(teacher_logits.masked_fill(~mask, -1e4), dim=-1)
        student_log_probs = torch.log_softmax(logits.masked_fill(~mask, -1e4), dim=-1)
        kd = -(teacher_probs * student_log_probs * mask).sum() / mask.sum()
        ce = soft_ce_loss(logits, target, mask)
        loss = kd + args.ce_weight * ce
        window = min(args.grad_accum, steps_per_epoch - (step // args.grad_accum) * args.grad_accum)
        (loss / window).backward()
        total_loss += float(loss.detach())
        if (step + 1) % args.grad_accum == 0 or step + 1 == steps_per_epoch:
            torch.nn.utils.clip_grad_norm_(encoder_params + head_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            if updates % 10 == 0:
                print(f"QAT update={updates} step={step+1}/{steps_per_epoch} loss={total_loss/(step+1):.6f} "
                      f"cuda={torch.cuda.memory_allocated()/2**30:.2f}GiB", flush=True)
    student.eval()
    student_logits = predict(student, tok, val, device)
    temperature = float(cfg.get("temperature", [1.0])[0])
    validation = quality(student_logits, teacher_val, val_labels, temperature=temperature)
    # Materialize the fake-quant weights before the ordinary v7 checkpoint/ONNX pipeline.
    with torch.no_grad():
        for module in quantized_modules:
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    save_checkpoint(student, tok, cfg, str(args.out))
    result = {"train_items": len(train), "val_items": len(val), "micro_batch": args.micro_batch,
              "grad_accum": args.grad_accum, "encoder_lr": args.encoder_lr, "head_lr": args.head_lr,
              "ce_weight": args.ce_weight,
              "updates": updates, "mean_loss": total_loss / steps_per_epoch,
              "fake_quant_modules": len(quantized_modules), "validation": validation,
              "teacher_cache": str(args.teacher_cache)}
    (args.out / "qat-metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print("QAT_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
