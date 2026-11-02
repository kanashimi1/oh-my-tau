#!/usr/bin/env python3
"""Head-only Laya context-policy training with validation selection and exact resume."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from laya.calibrate import fit_temperature_map
from laya.common import collate_items, uses_parallel_layout
from laya.train import (TrainConfig, _forward, calibration_records, draw_option_order, encode_item,
                        evaluate_records, items_from_rows, load_checkpoint, optimizer_updates,
                        read_data, save_checkpoint, soft_ce_loss)

ACTIONS = ("noop", "clm_edit", "rlm_offload", "rlm_recall")


def atomic_torch(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def metrics(records):
    result = evaluate_records(records, 1.0, None)
    confusion = {a: {b: 0 for b in ACTIONS} for a in ACTIONS}
    f1 = []
    decoded = [(int(logits[:len(ACTIONS)].argmax()), max(range(len(ACTIONS)), key=target.__getitem__))
               for _qtype, logits, target, _k in records]
    for predicted, expected in decoded:
        confusion[ACTIONS[expected]][ACTIONS[predicted]] += 1
    for index, action in enumerate(ACTIONS):
        tp = fp = fn = 0
        for predicted, expected in decoded:
            tp += predicted == index and expected == index
            fp += predicted == index and expected != index
            fn += predicted != index and expected == index
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1.append(2 * precision * recall / max(1e-12, precision + recall))
        result[f"f1_{action}"] = f1[-1]
        result[f"recall_{action}"] = recall
    result["macro_f1"] = sum(f1) / len(f1)
    result["confusion"] = confusion
    return result


def head_state(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items() if not key.startswith("encoder.")}


def load_head(model, state):
    incompatible = model.load_state_dict(state, strict=False)
    assert all(key.startswith("encoder.") for key in incompatible.missing_keys), incompatible.missing_keys
    assert not incompatible.unexpected_keys, incompatible.unexpected_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("/home/kaneki/fleet-split/laya-models/typed-decisions"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--class-balanced", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--unfreeze-encoder", action="store_true",
                        help="fine-tune encoder too (encoder_lr) so numeric state lines become readable")
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    args = parser.parse_args()
    resume_path = args.out / "training-state.pt"
    if args.resume and not resume_path.exists():
        raise FileNotFoundError(resume_path)
    if not args.resume and args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite recovery training: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(b"".join((args.data / f"{s}.jsonl").read_bytes() for s in ("train", "val", "test"))).hexdigest()
    config = TrainConfig(epochs=args.epochs, micro_batch=args.batch_size, grad_accum=args.grad_accum,
                         head_lr=args.lr, loss="soft-ce", freeze_encoder=not args.unfreeze_encoder,
                         encoder_lr=args.encoder_lr if args.unfreeze_encoder else 0.0, shuffle_options=("choice",),
                         gradient_checkpointing=False, seed=41873, log_every=100)
    config.validate()
    protocol = {"digest": digest, "config": dataclasses.asdict(config), "base": str(args.base),
                "class_balanced": args.class_balanced}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = load_checkpoint(str(args.base))
    max_len = cfg.get("max_len", 1024)
    head_max_len = cfg.get("head_max_len", 256)
    parallel = uses_parallel_layout(cfg)
    items = {}
    for split in ("train", "val", "test"):
        rows = read_data(str(args.data / f"{split}.jsonl"))
        items[split], skipped = items_from_rows(tok, rows, max_len, head_max_len)
        if not items[split] or skipped:
            raise ValueError(f"{split}: items={len(items[split])}, skipped={skipped}")
        for item in items[split]:
            assert item["k"] == len(ACTIONS)
    class_weights = None
    if args.class_balanced:
        counts = torch.zeros(len(ACTIONS))
        for item in items["train"]:
            counts[max(range(len(ACTIONS)), key=item["target"].__getitem__)] += 1
        class_weights = counts.sum().sqrt() / counts.clamp_min(1).sqrt()
        class_weights = (class_weights / class_weights.mean()).to(device)
        print("RECOVERY_CLASS_WEIGHTS " + json.dumps(dict(zip(ACTIONS, class_weights.cpu().tolist()))), flush=True)
    model.to(device)
    if device.type == "cuda":
        model.encoder.gradient_checkpointing_disable()
    model.head_checkpointing = False
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(args.unfreeze_encoder)
    model.encoder.eval()
    head_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.") and p.requires_grad]
    enc_params = [p for name, p in model.named_parameters() if name.startswith("encoder.") and p.requires_grad] \
        if args.unfreeze_encoder else []
    param_groups = ([{"params": head_params, "lr": config.head_lr},
                     {"params": enc_params, "lr": config.encoder_lr}]
                    if args.unfreeze_encoder else head_params)
    parameters = head_params + enc_params  # flat, for grad clipping
    optimizer = torch.optim.AdamW(param_groups, lr=config.head_lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optimizer_updates(len(items["train"]), config.micro_batch, config.grad_accum, config.epochs),
        eta_min=config.min_lr)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    order_rng = random.Random(config.seed)
    best_score = (-1.0, -math.inf)
    best_head = head_state(model)
    best_encoder = {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()} if args.unfreeze_encoder else None
    best_epoch = 0
    history = []
    start_epoch = 0
    if args.resume:
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved["protocol"] != protocol:
            raise ValueError("Recovery resume protocol mismatch")
        load_head(model, saved["head"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if scaler is not None and saved["scaler"] is not None:
            scaler.load_state_dict(saved["scaler"])
        order_rng.setstate(saved["option_rng"])
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        best_score, best_head, best_epoch = saved["best_score"], saved["best_head"], saved["best_epoch"]
        history, start_epoch = saved["history"], saved["epoch"] + 1
    else:
        base = metrics(calibration_records(model, tok, items["val"], device, max_len, head_max_len, parallel=parallel))
        atomic_json({"validation": base}, args.out / "base-validation.json")

    for epoch in range(start_epoch, config.epochs):
        model.train()
        model.encoder.eval()
        epoch_items = list(items["train"])
        random.Random(config.seed + epoch).shuffle(epoch_items)
        optimizer.zero_grad(set_to_none=True)
        total, steps = 0.0, 0
        steps_per_epoch = math.ceil(len(epoch_items) / config.micro_batch)
        for start in range(0, len(epoch_items), config.micro_batch):
            chunk = [encode_item(tok, item, max_len, head_max_len,
                                 draw_option_order(item, order_rng, config.shuffle_options), parallel)
                     for item in epoch_items[start:start + config.micro_batch]]
            batch = collate_items([chunk], tok.pad_token_id)
            logits = _forward(model, batch, device, device.type == "cuda", not args.unfreeze_encoder)
            mask = batch["marker_mask"].to(device)
            target = batch["target"].to(device)
            if class_weights is None:
                loss = soft_ce_loss(logits, target, mask)
            else:
                logp = torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
                per_item = -(target * logp * mask).sum(-1)
                loss = (per_item * (target * class_weights).sum(-1)).mean()
            window_start = (steps // config.grad_accum) * config.grad_accum
            window_size = min(config.grad_accum, steps_per_epoch - window_start)
            scaled = loss / window_size
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            steps += 1
            if steps % config.grad_accum == 0 or start + config.micro_batch >= len(epoch_items):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
            if not math.isfinite(total):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}")
        validation = metrics(calibration_records(model, tok, items["val"], device, max_len, head_max_len, parallel=parallel))
        mean_loss = total / max(1, steps)
        history.append({"epoch": epoch + 1, "train_loss": mean_loss, "validation": validation})
        score = (validation["macro_f1"], -validation["loss"])
        if score > best_score:
            best_score, best_head, best_epoch = score, head_state(model), epoch + 1
            if best_encoder is not None:
                best_encoder = {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
        atomic_torch({"epoch": epoch, "protocol": protocol, "head": head_state(model),
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "scaler": scaler.state_dict() if scaler is not None else None,
                      "option_rng": order_rng.getstate(), "python_rng": random.getstate(),
                      "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                      "best_score": best_score, "best_head": best_head, "best_epoch": best_epoch,
                      "history": history}, resume_path)
        print(f"RECOVERY_EPOCH {epoch + 1} loss={mean_loss:.6f} val_macro_f1={validation['macro_f1']:.6f} "
              f"val_acc={validation['accuracy']:.6f} recall={validation['recall_rlm_recall']:.6f}", flush=True)

    load_head(model, best_head)
    if best_encoder is not None:
        model.encoder.load_state_dict(best_encoder)
    model.eval()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    validation_records = calibration_records(model, tok, items["val"], device, max_len, head_max_len, parallel=parallel)
    fitted = fit_temperature_map(validation_records)
    out_cfg = dict(cfg, max_len=max_len, head_max_len=head_max_len,
                   temperature=fitted["temperature"], temperature_by_options=fitted["temperature_by_options"],
                   training={"recovery_best_epoch": best_epoch, "freeze_encoder": config.freeze_encoder,
                             "validation_selection": "macro_f1 then negative NLL"})
    save_checkpoint(model, tok, out_cfg, str(args.out / "ckpt"))
    # Test is touched once, after checkpoint and calibration were fixed on validation.
    test_records = calibration_records(model, tok, items["test"], device, max_len, head_max_len, parallel=parallel)
    base_model, base_tok, base_cfg = load_checkpoint(str(args.base))
    base_model.to(device).eval()
    base_test_records = calibration_records(base_model, base_tok, items["test"], device, max_len, head_max_len, parallel=parallel)
    result = {"best_epoch": best_epoch, "history": history, "class_balanced": args.class_balanced,
              "validation": evaluate_records(validation_records, fitted["temperature"], fitted["temperature_by_options"]),
              "test": metrics(test_records), "base_test": metrics(base_test_records),
              "protocol": "Frozen ModernBERT/ONNX-compatible encoder; option-order augmentation; validation-only epoch and temperature selection; unseen paraphrased test evaluated once."}
    atomic_json(result, args.out / "metrics.json")
    resume_path.unlink(missing_ok=True)
    print("RECOVERY_TRAINED " + json.dumps({"best_epoch": best_epoch, "test": result["test"],
                                            "base_test": result["base_test"]}), flush=True)


if __name__ == "__main__":
    main()
