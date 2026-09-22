#!/usr/bin/env python3
"""Train NIYAH reference Transformer on explicit record-level train/validation files."""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch

from niyah_ref import Config, NiyahLM, batch_from, evaluate, load_jsonl, save_checkpoint


def to_tensor(path: Path) -> torch.Tensor:
    return torch.tensor(list(load_jsonl(path)), dtype=torch.long)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--validation", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--context", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=768)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1448)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-iters", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--device")
    args = ap.parse_args()

    if args.steps <= 0 or args.batch <= 0 or args.accum <= 0:
        raise SystemExit("steps, batch and accum must be > 0")
    if args.d_model % args.heads != 0:
        raise SystemExit("d-model must be divisible by heads")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = Config(
        context=args.context,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )

    train_data = to_tensor(args.train)
    val_data = to_tensor(args.validation)
    if train_data.numel() <= cfg.context + 1:
        raise SystemExit("training split is too small for context")
    if val_data.numel() <= cfg.context + 1:
        raise SystemExit("validation split is too small for context")

    model = NiyahLM(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    params = sum(p.numel() for p in model.parameters())
    print(f"DEVICE={device}")
    print(f"PARAMETERS={params}")
    print(f"TRAIN_BYTES={train_data.numel()}")
    print(f"VALIDATION_BYTES={val_data.numel()}")

    initial_val = evaluate(model, val_data, args.batch, cfg.context, device, args.eval_iters)
    print(f"STEP=0 VAL_LOSS={initial_val:.6f} VAL_PPL={math.exp(min(20.0, initial_val)):.6f}")

    model.train()
    started = time.time()
    final_val = initial_val

    for step in range(1, args.steps + 1):
        progress = (step - 1) / max(1, args.steps - 1)
        lr = args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(args.accum):
            x, y = batch_from(train_data, args.batch, cfg.context, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                _, loss = model(x, y)
                scaled_loss = loss / args.accum
            scaler.scale(scaled_loss).backward()
            train_loss += float(loss.item())

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.log_every == 0:
            print(
                f"STEP={step} TRAIN_LOSS={train_loss / args.accum:.6f} "
                f"LR={lr:.8g} GRAD_NORM={float(grad_norm):.6f} "
                f"ELAPSED={time.time() - started:.2f}"
            )

        if step % args.eval_every == 0 or step == args.steps:
            final_val = evaluate(model, val_data, args.batch, cfg.context, device, args.eval_iters)
            print(f"STEP={step} VAL_LOSS={final_val:.6f} VAL_PPL={math.exp(min(20.0, final_val)):.6f}")

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(args.out, model, optimizer, step, args.seed)
            print(f"CHECKPOINT={args.out} STEP={step}")

    print(f"INITIAL_VAL_LOSS={initial_val:.6f}")
    print(f"FINAL_VAL_LOSS={final_val:.6f}")
    print(f"VAL_DELTA={final_val - initial_val:.6f}")
    print("TRAINING_COMPLETE=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
