# model-evals

Compare local model candidates via [promptfoo](https://promptfoo.dev) — code-output reasoning,
bug-spotting, PR review, and finding-classification evals against models served through a local
LiteLLM gateway (plus a couple of custom encoder providers). Results and cross-model summaries
live in `results/`.

> **Note:** this repo is for my own personal use, so config files reference machines on my home
> LAN by hostname (`rocky`, `rusty`, `bitey`, `sparky`) and private addresses like `10.0.0.2`.
> Those names won't resolve anywhere but my network.

## Structure

```
evals/
  cruxeval/       Code-output reasoning (pass@1)
  pr-review/      LLM-as-judge: does the model catch real bugs in real diffs?
  classifier/     Finding re-classification: chat models vs GLiNER2/Laya encoders
  code-gen/       Deterministic + rubric: instruct mode, thinking disabled
  bugfinding/     QuixBugs bug-spotting: execute the model's own fix against the test suite
benchmarks/       Standalone scripts: long-context needle retrieval, gateway load testing
providers/        Local gateway routes + custom GLiNER2/Laya providers
results/          Raw promptfoo output + per-session summaries (SUMMARY.md, MASTER.md)
scripts/          Eval/training helpers (quixbugs runner, corpus builders, judges)
data/             Eval inputs, vendored corpora, classification harvests (gitignored)
```

## Setup

```bash
npm i -g promptfoo
export ANTHROPIC_API_KEY=...   # judge model for llm-rubric assertions
export GATEWAY_URL=http://localhost:4000/v1   # or set in promptfoo.yaml
```

## Running

```bash
# Single eval
promptfoo eval -c evals/pr-review/promptfoo.yaml

# All evals
promptfoo eval

# Side-by-side results in browser
promptfoo view
```

## evals

- `cruxeval/` — code-output reasoning: `promptfoo-nemotron-full.yaml` (nemotron, 50 tests), `promptfoo-qwen.yaml` (qwen dense 27B, think/no-think/code)
- `classifier/` — finding re-classification: `promptfoo-qwen.yaml`, `promptfoo-nemotron.yaml`, `promptfoo-gemma4-*.yaml`, plus GLiNER2/Laya encoder variants
- `pr-review/` — LLM-as-judge over real PR diffs from my own repos (see below)
- `bugfinding/` — QuixBugs bug-spotting (self-contained)
- `code-gen/` — deterministic + rubric assertions, thinking disabled

## Bug-finding eval (QuixBugs)

Deterministic, self-contained bug-spotting eval. For each of 23 QuixBugs programs with
an known single-line bug, show the model the buggy source plus one concrete failing
case, ask it to output the corrected function, then **execute the model's own fix**
against the program's full test suite.

```bash
python3 scripts/quixbugs_runner.py --models nemotron qwen qwen-instruct nemotron-instruct
# data vendored in evals/bugfinding/quixbugs/ (QuixBugs BSD-3-Clause, LICENSE included)
```

Grading: `fixed` = model's function passes every test the correct version passes;
`part` = passes more than the buggy version; `notfixed` = no better than buggy.
No LLM judge, no cloud API — only the local gateway.

## Adding PR review test cases

Collect real diffs from repos where you know what the correct review looks like:

```bash
# From a local branch
git -C ~/Code/avenor diff main...your-branch > evals/pr-review/testdata/avenor-your-branch.diff

# From a closed GitHub PR
gh pr diff 42 --repo sdougbrown/avenor > evals/pr-review/testdata/avenor-pr-42.diff
```

Then add a test case to `evals/pr-review/promptfoo.yaml` with `known_issues` listing bugs you
already know exist in that diff. The judge checks whether the model found them.

## Relationship to longe

- **model-evals (this repo)**: which raw model produces better output? Single/few-turn, model comparison.
- **longe**: does swapping the model improve the full agentic loop? Multi-turn, agent evaluation.

The longe `code-review.yaml` schema (issues_spotted, false_positives, mitigations_suggested)
informed the rubric criteria used in `evals/pr-review/`.
