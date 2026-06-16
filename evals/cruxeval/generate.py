#!/usr/bin/env python3
"""
Generate CRUXEval-O promptfoo test cases from cruxeval-org/cruxeval (HuggingFace).

CRUXEval-O: given a Python function + input, predict the return value.
Self-contained — no repo context needed. Good for raw code reasoning comparison.

Usage:
    pip install datasets
    python3 generate.py            # 50 samples, seed 42
    python3 generate.py --n 100 --seed 0

Output: tests.yaml (loaded by promptfoo.yaml via `tests: file://tests.yaml`)

Leaderboard context (pass@1, ~800 samples):
  GPT-4o:         ~74% O / ~72% I
  claude-3.5:     ~73% O
  Qwen2.5-72B:    ~71% O
  Llama-3.1-70B:  ~64% O
"""

import argparse
import random
import sys
import yaml

def load_dataset_samples(n, seed):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets", file=sys.stderr)
        sys.exit(1)

    ds = load_dataset("cruxeval-org/cruxeval", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    return [ds[i] for i in sorted(indices)]


def make_test(sample):
    code = sample["code"]
    inp = sample["input"]
    expected = sample["output"]
    sample_id = sample.get("id", "unknown")

    # Pure JS assertion — no subprocess (require('child_process') is unavailable
    # in promptfoo's sandbox). Handles the common repr format mismatches:
    #   - markdown fences / inline backticks
    #   - multi-line preamble (take last non-empty line)
    #   - double-quote vs single-quote throughout (Python repr style)
    assertion_js = (
        "(() => {"
        " let got = output.trim()"
        "   .replace(/^```[\\w]*\\n?/m,'').replace(/\\n?```$/m,'').trim();"
        " const lines = got.split('\\n').map(l=>l.trim()).filter(Boolean);"
        " if (lines.length > 1) got = lines[lines.length-1];"
        " got = got.replace(/^`([^`]+)`$/,'$1').trim();"
        " const exp = context.vars.expected_output.trim();"
        " if (got === exp) return true;"
        " const nq = s => s.replace(/\"/g,\"'\");"
        " if (nq(got) === nq(exp)) return true;"
        " return false;"
        " })()"
    )
    return {
        "description": f"cruxeval-o: {sample_id}",
        "vars": {
            "code": code,
            "input": inp,
            "expected_output": expected,
        },
        "assert": [{"type": "javascript", "value": assertion_js}],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="tests.yaml")
    args = parser.parse_args()

    print(f"Loading {args.n} samples (seed={args.seed})...")
    samples = load_dataset_samples(args.n, args.seed)

    tests = [make_test(s) for s in samples]

    with open(args.output, "w") as f:
        yaml.dump(tests, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"Wrote {len(tests)} test cases to {args.output}")
    print("Next: promptfoo eval -c promptfoo.yaml")


if __name__ == "__main__":
    main()
