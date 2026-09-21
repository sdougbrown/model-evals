#!/usr/bin/env python3
"""Single-GPU RLCD fine-tune of laya (typed-decisions base) on the umpire corpus.

Adapted from the official notebook (laya_finetune_typed_decisions_2xT4_kaggle)
with DDP removed and per-epoch val evaluation + best-checkpoint tracking added.
Expects items built by build-laya-items.py.

Usage:
  ~/.venvs/laya-gpu/bin/python scripts/train-laya-umpire.py \
    --base ~/Models/laya/typed-decisions --data data/laya --out models/laya-umpire
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from laya.common import build_model, proper_reward  # noqa: E402


def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


@torch.no_grad()
def eval_ce(model, items, tok, device):
    """Mean soft CE and argmax accuracy on val items — selection uses accuracy."""
    model.eval()
    losses, correct = [], 0
    for b in range(0, len(items), 16):
        batch = collate_train_batch(items[b:b + 16], tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
        logits = logits.float()
        mask = batch["marker_mask"].to(device)
        ce = -(batch["target"].to(device) * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1)
        losses.extend(ce.tolist())
        amax = logits.argmax(-1)
        for i in range(logits.size(0)):
            k = int(mask[i].sum().item())
            # argmax over this question's option markers only
            top = torch.argmax(logits[i, :k]).item()
            correct += int(top == batch["label"][i].item())
    model.train()
    return sum(losses) / len(losses), correct / max(1, len(losses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path.home() / "Models/laya/typed-decisions")
    ap.add_argument("--data", type=Path, default=Path("data/laya"))
    ap.add_argument("--out", type=Path, default=Path("models/laya-umpire"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    with open(args.base / "rl_agent_config.json") as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = 4096
    cfg["max_len"] = 1024
    cfg["head_max_len"] = 256

    tok = AutoTokenizer.from_pretrained(str(args.base / "tokenizer"))
    model = build_model(cfg, encoder_dir=str(args.base / "encoder"))
    model.load_state_dict(load_file(str(args.base / "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()

    train_items = torch.load(args.data / "train_items.pt", weights_only=False)
    val_items = torch.load(args.data / "val_items.pt", weights_only=False)

    MICRO_BATCH, GRAD_ACCUM = args.micro_batch, args.grad_accum
    GROUP_SIZE, EPOCHS = 4, args.epochs
    LR_ENCODER, LR_HEAD = 2.5e-5, 1.0e-4
    SIGMA_START, SIGMA_END = 0.4, 0.1

    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": LR_ENCODER},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=0.01)
    total_updates = (len(train_items) // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    best_acc = -1.0
    t0 = time.time()
    for epoch in range(EPOCHS):
        random.seed(42 + epoch)
        random.shuffle(train_items)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (epoch / max(1, EPOCHS - 1))
        optimizer.zero_grad(set_to_none=True)
        accum_step, n_batches, epoch_loss = 0, 0, 0.0

        for b_idx in range(0, len(train_items), MICRO_BATCH):
            chunk = train_items[b_idx:b_idx + MICRO_BATCH]
            if not chunk:
                continue
            batch = collate_train_batch(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, act = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum_step += 1
            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(train_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1
            if n_batches % 25 == 0:
                print(f"  epoch {epoch+1}/{EPOCHS} step {n_batches} loss {loss.item()*GRAD_ACCUM:.4f} reward {r.mean().item():.3f}", flush=True)

        val_ce, val_acc = eval_ce(model, val_items, tok, device)
        print(f"=== epoch {epoch+1}/{EPOCHS} done in {time.time()-t0:.0f}s | train loss {epoch_loss/max(1,n_batches):.4f} | val CE {val_ce:.4f} | val acc {val_acc:.3f} ===", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            best_dir = args.out / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            save_model(model, tok, cfg, best_dir, val_ce)
            print(f"  new best acc ({val_acc:.3f}) saved to {best_dir}", flush=True)

    save_model(model, tok, cfg, args.out / "final", val_ce)
    print(f"done in {(time.time()-t0)/60:.1f} min | best val acc {best_acc:.3f} | final saved to {args.out/'final'}")


def save_model(model, tok, cfg, out_dir: Path, val_ce: float):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, str(out_dir / "model.safetensors"))
    model.encoder.config.save_pretrained(str(out_dir / "encoder"))
    tok.save_pretrained(str(out_dir / "tokenizer"))
    cfg = dict(cfg)
    cfg["fine_tuned"] = True
    cfg["model_name"] = "laya-umpire"
    cfg["val_ce"] = round(val_ce, 4)
    with open(out_dir / "rl_agent_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    model.train()


if __name__ == "__main__":
    main()
