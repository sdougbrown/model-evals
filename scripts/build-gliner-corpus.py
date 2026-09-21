#!/usr/bin/env python3
"""Build the GLiNER2 umpire-disposition training corpus from pr-reviewer data.

Joins the adjudicated classifications corpus (classifications/**/*.json) with
finding bodies from the pr-reviewer sqlite (state.db), plus optional sparky
extracts (discussion replies / current_code JSONL when present), and emits
train/val JSONL in the GLiNER2 tutorial-8 classification format.

Balancing: minority classes are oversampled (deterministic, seeded) so the
fixed/no_action majority doesn't dominate — the notes doc flags this as the
corpus's core imbalance problem.

Contamination guard: the 6 finding IDs in evals/classifier/tests.yaml are the
independent benchmark; they are excluded from train AND val.

Usage:
  python3 scripts/build-gliner-corpus.py \
    --data-dir data \
    --out-dir data/gliner \
    [--oversample-cap 4] [--val-frac 0.15] [--seed 13]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import re
import sqlite3
from pathlib import Path

# The classifier eval's 6 core finding IDs — benchmark, never train on them.
EVAL_FINDING_IDS = {
    3300055955, 3322114846, 3322011710, 3322114811, 3322011750, 3300533985,
}

CLASSIFICATION_LABELS = ["still_valid", "fixed", "obsolete", "superseded", "uncertain"]
DISPOSITION_LABELS = ["needs_action", "accepted", "informational", "no_action", "needs_review"]

# Rules lifted from evals/classifier/prompt.js — encoded into the training data
# as label_descriptions so the adapter learns the distinctions from text, and
# reused verbatim at inference.
CLASSIFICATION_DESCRIPTIONS = {
    "still_valid": "The finding describes an actual issue that still exists in the current code.",
    "fixed": "The finding was valid but the issue has been resolved.",
    "obsolete": "The finding is no longer relevant due to code changes.",
    "superseded": "The exact prior finding is no longer literally accurate, but a closely related issue remains.",
    "uncertain": "Not enough information to determine.",
}
DISPOSITION_DESCRIPTIONS = {
    "needs_action": "A live issue should be surfaced for action.",
    "accepted": "A live issue was intentionally accepted, deferred, tracked elsewhere, or rejected as not worth changing. Accepted is only ever a disposition, never a classification.",
    "informational": "Useful context, but not something a re-review should push as requiring action.",
    "no_action": "Fixed or obsolete; keep only for audit trail.",
    "needs_review": "A human should decide because evidence is ambiguous, low-confidence, or contradictory.",
}


def _is_bot_reply(reply: dict) -> bool:
    """Bot self-replies reflect prior bot state, not human judgment — the eval
    prompt ignores [AUTO] replies; training inputs should too."""
    author = reply.get("author") or ""
    body = reply.get("body") or ""
    return bool(re.search(r"umpire|github-actions|\bbot\b", author, re.I)
                or body.startswith("`👾 AI Agent`"))


def load_replies(data_dir: Path) -> dict[int, list[dict]]:
    """replies.jsonl — sparky extract rows are per-thread-group with a
    comment_github_ids array; also accept the per-comment row shape."""
    replies_by_comment: dict[int, list[dict]] = {}
    f = data_dir / "replies.jsonl"
    if not f.exists():
        return replies_by_comment
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        replies = [r for r in rec.get("replies", []) if not _is_bot_reply(r)]
        ids = rec.get("comment_github_ids") or (
            [rec["comment_github_id"]] if rec.get("comment_github_id") else []
        )
        for cid in ids:
            replies_by_comment[cid] = replies
    return replies_by_comment


def _truncate_code(code: str, limit: int = 2000) -> str:
    # DeBERTa attention is quadratic in subword length: a 9K-char input was
    # ~40GB of attention tensors and OOM-killed the first long-context run.
    # ~3K chars total ≈ 1.7K subwords ≈ ~6GB peak at batch 2. 2026-09-18.
    if len(code) <= limit:
        return code
    head = limit * 2 // 3
    return code[:head] + "\n...[truncated]...\n" + code[-(limit - head):]


def _truncate_input(text: str, limit: int = 3000) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return text[:head] + "\n...[truncated]...\n" + text[-(limit - head):]


def load_code(data_dir: Path) -> dict[int, str]:
    """current_code.jsonl — one row per (repo, path, head_sha) with a
    comment_github_ids array; also accept per-comment rows."""
    code_by_comment: dict[int, str] = {}
    f = data_dir / "current_code.jsonl"
    if not f.exists():
        f = data_dir / "code_context.jsonl"  # first-draft name
    if not f.exists():
        return code_by_comment
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        content = _truncate_code(rec.get("content") or "")
        ids = rec.get("comment_github_ids") or (
            [rec["comment_github_id"]] if rec.get("comment_github_id") else []
        )
        for cid in ids:
            code_by_comment[cid] = content
    return code_by_comment


def load_corpus(data_dir: Path) -> list[dict]:
    db = sqlite3.connect(data_dir / "state.db")
    bodies = {
        row[0]: {"body": row[1], "path": row[2], "repo": row[3], "pr": row[4]}
        for row in db.execute(
            "SELECT comment_github_id, body, path, repo, pr_number FROM pr_inline_comments"
        )
    }

    replies_by_comment = load_replies(data_dir)
    code_by_comment = load_code(data_dir)
    stats = collections.Counter()
    stats["reply_threads"] = len(replies_by_comment)
    stats["code_files"] = len(code_by_comment)
    examples: list[dict] = []
    for path in sorted(glob.glob(str(data_dir / "classifications" / "**" / "*.json"), recursive=True)):
        for entry in json.loads(Path(path).read_text()).get("entries", []):
            cid = entry.get("comment_id")
            cls = entry.get("classification", {})
            validation = cls.get("validation", {})
            label = cls.get("classification")
            disposition = cls.get("disposition")
            if not validation.get("valid") or label not in CLASSIFICATION_LABELS or disposition not in DISPOSITION_LABELS:
                stats["skipped_invalid"] += 1
                continue
            if cid in EVAL_FINDING_IDS:
                stats["skipped_eval_benchmark"] += 1
                continue
            src = bodies.get(cid)
            if not src or not src["body"]:
                stats["skipped_no_body"] += 1
                continue

            parts = [f"<finding>\n{src['body']}\n</finding>"]
            code = code_by_comment.get(cid)
            if code:
                parts.append(f"<current_code>\n{code}\n</current_code>")
            replies = replies_by_comment.get(cid, [])
            if replies:
                lines = [f"[{r.get('author', '?')}]: {r.get('body', '')}" for r in replies]
                parts.append("<discussion_replies>\n" + "\n".join(lines) + "\n</discussion_replies>")
            has_replies = bool(replies)

            examples.append({
                "comment_id": cid,
                "input": _truncate_input("\n\n".join(parts)),
                "classification": label,
                "disposition": disposition,
                "has_replies": has_replies,
                "confidence": cls.get("confidence"),
            })
            stats["loaded"] += 1
            stats[f"class_{label}"] += 1
            stats[f"disp_{disposition}"] += 1
            if has_replies:
                stats["with_replies"] += 1
    print("load stats:", dict(stats))
    return examples


def balance_and_split(
    examples: list[dict], val_frac: float, oversample_cap: int, seed: int
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_pair: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for e in examples:
        by_pair[(e["classification"], e["disposition"])].append(e)

    train, val = [], []
    for pair, items in sorted(by_pair.items()):
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_frac)) if len(items) >= 4 else 0
        val.extend(items[:n_val])
        pool = items[n_val:]
        # Oversample minorities toward the largest pair's train size, capped.
        train.extend(pool)
    max_train = max(
        len([e for e in train if (e["classification"], e["disposition"]) == p])
        for p in by_pair
    )
    target = min(max_train, max(1, max_train // oversample_cap) * oversample_cap)
    balanced: list[dict] = []
    for pair in by_pair:
        pool = [e for e in train if (e["classification"], e["disposition"]) == pair]
        if not pool:
            continue
        repeats = target // len(pool)
        remainder = target % len(pool)
        balanced.extend(pool * repeats + pool[:remainder])
    rng.shuffle(balanced)
    return balanced, val


def to_gliner_jsonl(examples: list[dict]) -> str:
    lines = []
    for e in examples:
        lines.append(json.dumps({
            "input": e["input"],
            "output": {
                "classifications": [
                    {
                        "task": "classification",
                        "labels": CLASSIFICATION_LABELS,
                        # processor.py does true_label.copy() — must be a list,
                        # not the string the tutorial-8 table suggests
                        "true_label": [e["classification"]],
                        "label_descriptions": CLASSIFICATION_DESCRIPTIONS,
                    },
                    {
                        "task": "disposition",
                        "labels": DISPOSITION_LABELS,
                        "true_label": [e["disposition"]],
                        "label_descriptions": DISPOSITION_DESCRIPTIONS,
                    },
                ]
            },
        }))
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/gliner"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--oversample-cap", type=int, default=4,
                    help="minority pairs are oversampled up to 1/cap of the largest pair")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    examples = load_corpus(args.data_dir)
    if not examples:
        raise SystemExit("no usable examples — check data dir layout")
    train, val = balance_and_split(examples, args.val_frac, args.oversample_cap, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train.jsonl").write_text(to_gliner_jsonl(train))
    (args.out_dir / "val.jsonl").write_text(to_gliner_jsonl(val))
    (args.out_dir / "corpus-meta.json").write_text(json.dumps({
        "total": len(examples),
        "train": len(train),
        "val": len(val),
        "train_pairs": {"|".join(k): v for k, v in collections.Counter((e["classification"], e["disposition"]) for e in train).items()},
        "val_pairs": {"|".join(k): v for k, v in collections.Counter((e["classification"], e["disposition"]) for e in val).items()},
        "oversample_cap": args.oversample_cap,
        "seed": args.seed,
    }, indent=2))
    print(f"train={len(train)}  val={len(val)}  ->  {args.out_dir}/{{train,val}}.jsonl")
    print("val pairs:", dict(collections.Counter((e["classification"], e["disposition"]) for e in val)))


if __name__ == "__main__":
    main()
