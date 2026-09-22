#!/usr/bin/env python3
"""NIYAH reference causal Transformer trainer.

Purpose: provide a small, auditable training implementation that is independent of
external pretrained weights. It is a reference path for dataset/checkpoint/model
contracts; the native C11 runtime remains the deployment target.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 256
    context: int = 256
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    d_ff: int = 768
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class CausalAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = CausalAttention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.ff(self.n2(x))
        return x


class NiyahLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.context, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = ids.shape
        if t > self.cfg.context:
            raise ValueError(f"sequence {t} exceeds context {self.cfg.context}")
        p = torch.arange(t, device=ids.device)
        x = self.tok(ids) + self.pos(p)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(ids: list[int]) -> str:
    return bytes(int(x) & 0xFF for x in ids).decode("utf-8", errors="replace")


def load_jsonl(path: Path) -> bytes:
    chunks: list[bytes] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, 1):
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            instruction = str(rec.get("instruction", "")).strip()
            response = str(rec.get("response", "")).strip()
            if not instruction or not response:
                raise ValueError(f"missing instruction/response at line {line_no}")
            text = f"<|user|>\n{instruction}\n<|assistant|>\n{response}\n<|end|>\n"
            chunks.append(text.encode("utf-8"))
    if not chunks:
        raise ValueError("dataset contains no usable records")
    return b"".join(chunks)


def split_bytes(data: bytes, val_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    if not (0.0 < val_fraction < 0.5):
        raise ValueError("val_fraction must be in (0, 0.5)")
    cut = max(1, min(len(data) - 1, int(len(data) * (1.0 - val_fraction))))
    return torch.tensor(list(data[:cut]), dtype=torch.long), torch.tensor(list(data[cut:]), dtype=torch.long)


def batch_from(data: torch.Tensor, batch: int, context: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if data.numel() <= context + 1:
        raise ValueError("dataset split is too small for context")
    starts = torch.randint(0, data.numel() - context - 1, (batch,))
    x = torch.stack([data[i:i + context] for i in starts])
    y = torch.stack([data[i + 1:i + context + 1] for i in starts])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model: NiyahLM, data: torch.Tensor, batch: int, context: int, device: torch.device, iters: int) -> float:
    model.eval()
    values = []
    for _ in range(iters):
        x, y = batch_from(data, batch, context, device)
        _, loss = model(x, y)
        values.append(float(loss.item()))
    model.train()
    return sum(values) / len(values)


def save_checkpoint(path: Path, model: NiyahLM, optimizer: torch.optim.Optimizer, step: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "NIYAH_REF_PT1",
        "config": asdict(model.cfg),
        "step": step,
        "seed": seed,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[NiyahLM, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "NIYAH_REF_PT1":
        raise ValueError("unsupported checkpoint format")
    cfg = Config(**payload["config"])
    model = NiyahLM(cfg).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model, payload


def train(args) -> int:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = Config(context=args.context, d_model=args.d_model, n_heads=args.heads,
                 n_layers=args.layers, d_ff=args.d_ff, dropout=args.dropout)
    raw = load_jsonl(args.data)
    train_data, val_data = split_bytes(raw, args.val_fraction)

    start = 0
    if args.resume and args.out.exists():
        model, payload = load_checkpoint(args.out, device)
        if asdict(model.cfg) != asdict(cfg):
            raise ValueError("resume config differs from requested config")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        optimizer.load_state_dict(payload["optimizer"])
        start = int(payload["step"])
        torch.set_rng_state(payload["torch_rng"])
        if device.type == "cuda" and payload.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    else:
        model = NiyahLM(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    params = sum(p.numel() for p in model.parameters())
    print(f"DEVICE={device}")
    print(f"PARAMETERS={params}")
    print(f"TRAIN_BYTES={train_data.numel()}")
    print(f"VAL_BYTES={val_data.numel()}")

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    model.train()
    t0 = time.time()
    for step in range(start + 1, args.steps + 1):
        progress = (step - 1) / max(1, args.steps - 1)
        lr = args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.accum):
            x, y = batch_from(train_data, args.batch, cfg.context, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                _, loss = model(x, y)
                scaled_loss = loss / args.accum
            scaler.scale(scaled_loss).backward()
            total += float(loss.item())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"STEP={step} TRAIN_LOSS={total / args.accum:.6f} LR={lr:.8g} ELAPSED={elapsed:.2f}")
        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, val_data, args.batch, cfg.context, device, args.eval_iters)
            print(f"STEP={step} VAL_LOSS={val:.6f} VAL_PPL={math.exp(min(20.0, val)):.6f}")
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(args.out, model, optimizer, step, args.seed)
            print(f"CHECKPOINT={args.out} STEP={step}")
    return 0


@torch.no_grad()
def generate(args) -> int:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_checkpoint(args.model, device)
    model.eval()
    ids = encode(args.prompt)
    if not ids:
        ids = [10]
    for _ in range(args.tokens):
        window = ids[-model.cfg.context:]
        x = torch.tensor([window], dtype=torch.long, device=device)
        logits, _ = model(x)
        logits = logits[0, -1] / max(args.temperature, 1e-5)
        if args.top_k > 0:
            k = min(args.top_k, logits.numel())
            threshold = torch.topk(logits, k).values[-1]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        if args.temperature <= 0:
            nxt = int(torch.argmax(logits).item())
        else:
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
        ids.append(nxt)
    print(decode(ids))
    print(f"\nMODEL_STEP={payload['step']}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--accum", type=int, default=1)
    t.add_argument("--context", type=int, default=256)
    t.add_argument("--d-model", type=int, default=192)
    t.add_argument("--heads", type=int, default=6)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--d-ff", type=int, default=768)
    t.add_argument("--dropout", type=float, default=0.0)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--weight-decay", type=float, default=0.1)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=1448)
    t.add_argument("--val-fraction", type=float, default=0.05)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--eval-every", type=int, default=100)
    t.add_argument("--eval-iters", type=int, default=10)
    t.add_argument("--save-every", type=int, default=100)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--device")

    g = sub.add_parser("generate")
    g.add_argument("--model", type=Path, required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--tokens", type=int, default=128)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=40)
    g.add_argument("--device")
    return p


def main() -> int:
    args = parser().parse_args()
    return train(args) if args.cmd == "train" else generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
