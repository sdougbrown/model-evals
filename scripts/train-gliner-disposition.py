#!/usr/bin/env python3
"""LoRA-train a gliner2.5 disposition/classification adapter on the umpire corpus.

Trains on data/gliner/train.jsonl (build with scripts/build-gliner-corpus.py),
evaluates per epoch on val.jsonl, saves an adapter (~MBs) that hot-swaps into
the running gliner2 service via model.load_adapter().

Default device is CPU (280M encoder, ~1.1k balanced examples — minutes to tens
of minutes). GLINER_DEVICE=cuda uses the UTIL tier slice when free.

Usage:
  ~/.venvs/gliner2/bin/python scripts/train-gliner-disposition.py \
    --base ~/Models/gliner2.5-multi-v1 --data data/gliner --out models/gliner2-umpire-adapter
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from gliner2 import AutoExtractor
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path.home() / "Models/gliner2.5-multi-v1")
    ap.add_argument("--data", type=Path, default=Path("data/gliner"))
    ap.add_argument("--out", type=Path, default=Path("models/gliner2-umpire-adapter"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    args = ap.parse_args()

    device = os.environ.get("GLINER_DEVICE", "cpu")
    fp16 = device == "cuda"

    model = AutoExtractor.from_pretrained(str(args.base), map_location=device)
    config = TrainingConfig(
        output_dir=str(args.out),
        experiment_name="umpire-disposition",
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 8 // args.batch_size),  # effective batch 8
        encoder_lr=1e-5,
        task_lr=5e-4,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=float(2 * args.lora_r),
        lora_dropout=0.05,
        # encoder + heads; the classification heads are where disposition lives
        lora_target_modules=["encoder", "span_rep", "classifier"],
        save_adapter_only=True,
        eval_strategy="epoch",
        logging_steps=10,
        fp16=fp16,
    )
    trainer = ExtractorTrainer(model=model, config=config)
    trainer.train(
        train_data=str(args.data / "train.jsonl"),
        eval_data=str(args.data / "val.jsonl"),
    )
    print(f"adapter saved under {args.out}/final/")


if __name__ == "__main__":
    main()
