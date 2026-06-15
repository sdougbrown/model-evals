# model-evals

Compare local model candidates (currently: qwen-moe vs nex-mini) via [promptfoo](https://promptfoo.dev).

## Structure

```
evals/
  pr-review/      LLM-as-judge: does the model catch real bugs in real diffs?
  code-gen/       Deterministic + rubric: instruct mode, thinking disabled
providers/
  local.yaml      Gateway routes for all local model variants
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
