#!/usr/bin/env python3
"""
Build a WT prose-revision training corpus (~400 examples) from local sources:

  KEEP/positive : clean prose passages extracted by prose-extract (K8s + Rails docs)
  REWRITE/neg   : AI-slop passages from ~/Code/_notes markdown, PLUS clean known-good
                  passages that WT-lint flags with a real prose defect.

Labeling is deterministic via WT-lint (writetighter CLI) under a strict materiality gate.
Slop filtering (code/command/config removal) lives in slop_filter.py.

Usage:
  python3 build_prose_corpus.py --good k8s_passages.jsonl rails_passages.jsonl --out prose_corpus.jsonl
"""
import json, os, re, sys, subprocess, collections, random, argparse

# Reuse the tested code-filter logic
from slop_filter import is_code_line, is_prose_passage

WT = os.path.expanduser("~/Code/writetighter/writetighter")
N_KEEP_TARGET = 200
N_REWRITE_TARGET = 200
GOOD_SAMPLE = 2000
SLOP_SAMPLE = 500

SLOP_SKIP = {"AGENTS.md", "CLAUDE.md", "README.md", "CONTRIBUTING.md", "CODEOWNERS"}
SLOP_SKIP_PREFIX = ("model-", "cmake-", "llm-")

# Materiality gate: a passage is REWRITE iff it triggers a strong signal.
NOUN_STACK_MIN = 5
STRONG_RULES = {"CORE.BANNED_MODAL", "CORE.CONTRACTION", "CORE.TERM_DISCOURAGED",
                "CORE.LATIN_ABBREV", "CORE.GERUND_OPENER"}


def lint_text(text):
    r = subprocess.run([WT, "lint", "--stdin", "--kind", "description", "--format", "json"],
                       input=text.encode(), capture_output=True, timeout=30)
    try:
        d = json.loads(r.stdout)
    except Exception:
        return r.returncode, []
    return r.returncode, d.get("findings", [])


def is_rewrite(findings):
    for f in findings:
        rid = f.get("rule_id")
        sev = f.get("severity")
        ev = f.get("evidence") or ""
        if rid == "CORE.NOUN_STACK":
            m = re.search(r"noun stack \((\d+) content words\)", ev)
            if m and int(m.group(1)) >= NOUN_STACK_MIN:
                return True
            continue
        if rid in STRONG_RULES:
            return True
        if sev in ("warning", "error") and rid != "CORE.SENTENCE_LENGTH":
            return True
    return False


def load_good(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def load_slop(root):
    passages = []
    for dirpath, _, files in os.walk(root):
        if "/.git" in dirpath or "/archive" in dirpath:
            continue
        for fn in files:
            if not fn.endswith(".md") or fn in SLOP_SKIP or fn.startswith(SLOP_SKIP_PREFIX):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p) as f:
                    lines = f.read().splitlines()
            except Exception:
                continue
            buf = []
            for ln in lines:
                s = ln.strip()
                if not s or s.startswith(("#", "-", "*", "|", "```", ">", "1.", "[", "![")):
                    if len(" ".join(buf).split()) >= 8:
                        passages.append((os.path.relpath(p, root), " ".join(buf)))
                    buf = []
                    continue
                if is_code_line(s):
                    if buf and len(" ".join(buf).split()) >= 8:
                        passages.append((os.path.relpath(p, root), " ".join(buf)))
                    buf = []
                    continue
                buf.append(s)
                if len(" ".join(buf).split()) >= 50:
                    passages.append((os.path.relpath(p, root), " ".join(buf)))
                    buf = []
            if buf and len(" ".join(buf).split()) >= 8:
                passages.append((os.path.relpath(p, root), " ".join(buf)))
    filtered = []
    for rel, text in passages:
        if is_prose_passage(text.split()):
            filtered.append((rel, text))
    return filtered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--good", nargs="+", default=["/tmp/k8s_passages.jsonl", "/tmp/rails_passages.jsonl"])
    ap.add_argument("--slop-dir", default=os.path.expanduser("~/Code/_notes"))
    ap.add_argument("--out", default=os.path.expanduser("~/Code/_notes/ai/data/umpire-classification-corpus/prose_corpus.jsonl"))
    ap.add_argument("--seed", type=int, default=20260807)
    args = ap.parse_args()
    random.seed(args.seed)

    good = load_good(args.good)
    print("clean good passages:", len(good))

    keep_examples = []
    rewrite_examples = []
    good_sample = random.sample(good, min(GOOD_SAMPLE, len(good)))
    for i, g in enumerate(good_sample):
        text = (g.get("text") or "").strip()
        if not (8 <= len(text.split()) <= 60):
            continue
        rc, findings = lint_text(text)
        mat = is_rewrite(findings)
        rec = {"text": text, "provenance": {"kind": "known_good", "src": g.get("source"), "file": g.get("file")},
               "findings": findings, "private": False}
        if mat:
            rewrite_examples.append(rec)
        else:
            keep_examples.append(rec)
        if (i + 1) % 100 == 0:
            print("  linted", i + 1, "keep=", len(keep_examples), "rewrite=", len(rewrite_examples))
    print("after good docs: keep=", len(keep_examples), "rewrite(known-good flagged)=", len(rewrite_examples))

    slop = load_slop(args.slop_dir)
    print("slop passages extracted:", len(slop))
    slop_sample = random.sample(slop, min(SLOP_SAMPLE, len(slop)))
    for fpath, text in slop_sample:
        rc, findings = lint_text(text)
        mat = is_rewrite(findings)
        rec = {"text": text, "provenance": {"kind": "slop", "file": fpath}, "findings": findings, "private": True}
        if mat:
            rewrite_examples.append(rec)
        else:
            keep_examples.append(rec)
    print("after slop: keep=", len(keep_examples), "rewrite=", len(rewrite_examples))

    random.shuffle(keep_examples)
    random.shuffle(rewrite_examples)
    keep_examples = keep_examples[:N_KEEP_TARGET]
    rewrite_examples = rewrite_examples[:N_REWRITE_TARGET]

    rows = []
    for i, e in enumerate(keep_examples):
        e["id"] = "prose-keep-%04d" % i
        e["keep_class"] = True
        e["split"] = "train" if i % 8 else "holdout"
        rows.append(e)
    for i, e in enumerate(rewrite_examples):
        e["id"] = "prose-rewrite-%04d" % i
        e["keep_class"] = False
        e["split"] = "train" if i % 8 else "holdout"
        rows.append(e)
    random.shuffle(rows)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print()
    print("WROTE", os.path.abspath(args.out))
    print("total:", len(rows), "| keep:", sum(1 for r in rows if r["keep_class"]),
          "| rewrite:", sum(1 for r in rows if not r["keep_class"]))
    print("provenance: known_good=", sum(1 for r in rows if not r["private"]),
          "| slop(private)=", sum(1 for r in rows if r["private"]))
    print("split:", dict(collections.Counter(r["split"] for r in rows)))


if __name__ == "__main__":
    main()
