#!/usr/bin/env python3
"""Build laya training items from the umpire corpus (GLiNER JSONL → tokenized items).

Reads data/gliner/{train,val}.jsonl (same corpus as the GLiNER adapter — build
with build-gliner-corpus.py first), renders each finding through laya's
build_sequence against the two umpire questions, and emits .pt item files for
train-laya-umpire.py.

Question definitions mirror serving/gliner2/app.py (LAYA_*_Q) — that file is
canonical; keep the two in sync.

Usage: ~/.venvs/laya-gpu/bin/python scripts/build-laya-items.py --base ~/Models/laya/typed-decisions
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer


CLASSIFICATION_Q = {
    "type": "choice",
    "instructions": "Is the prior code-review finding still present in the current code? Code truth only; discussion replies indicate intent but do not prove code truth.",
    "criteria": {
        "still_valid": "the issue still exists in the current code",
        "fixed": "the finding was valid but has been resolved",
        "obsolete": "no longer relevant due to code changes",
        "superseded": "no longer literally accurate but a closely related issue remains",
        "uncertain": "not enough information to determine",
    },
}
DISPOSITION_Q = {
    "type": "choice",
    "instructions": "What workflow action does this finding need? Replies marked AUTO are mechanical bot state, not human judgment.",
    "criteria": {
        "needs_action": "a live issue should be surfaced for action in this PR",
        "accepted": "intentionally accepted, deferred, or tracked as follow-up in discussion",
        "informational": "useful context, intentionally low-action, no fix requested",
        "no_action": "fixed or obsolete; keep for audit trail only",
        "needs_review": "a human should decide; evidence ambiguous or contradictory",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path.home() / "Models/laya/typed-decisions")
    ap.add_argument("--data", type=Path, default=Path("data/gliner"))
    ap.add_argument("--out", type=Path, default=Path("data/laya"))
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=256)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path.home() / "Code" / "_wt"))  # not needed; laya is installed
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_sequence, render_options

    with open(args.base / "rl_agent_config.json") as f:
        cfg = json.load(f)
    _fix_tokenizer_config(str(args.base))
    tok = AutoTokenizer.from_pretrained(str(args.base / "tokenizer"))

    def build_item(text: str, q: dict, gold_label: str):
        crit = q["criteria"]
        keys = list(crit.keys())
        target = [1.0 if k == gold_label else 0.0 for k in keys]
        k_expected = len(render_options({"t": q["type"], "crit": crit}))
        seq, markers = build_sequence(
            tok, {"body": text},
            {"t": q["type"], "ins": q["instructions"], "crit": crit},
            args.max_len, args.head_max_len,
        )
        if len(markers) != k_expected:
            return None  # option markers got truncated out of the budget
        return {
            "ids": seq,
            "markers": markers,
            "qtype": 0,  # QTYPES["choice"] == 0
            "target": target,
            "label": target.index(max(target)),
        }

    for split in ("train", "val"):
        items, dropped = [], 0
        for line in (args.data / f"{split}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec["input"]
            for q, key in ((CLASSIFICATION_Q, "classification"), (DISPOSITION_Q, "disposition")):
                cls_task, dis_task = rec["output"]["classifications"]
                gold = cls_task["true_label"][0] if key == "classification" else dis_task["true_label"][0]
                it = build_item(text, q, gold)
                if it:
                    items.append(it)
                else:
                    dropped += 1
        args.out.mkdir(parents=True, exist_ok=True)
        torch.save(items, args.out / f"{split}_items.pt")
        print(f"{split}: {len(items)} items ({dropped} dropped for marker truncation)")


if __name__ == "__main__":
    main()
