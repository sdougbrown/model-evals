# Model eval results — 2026-08-14

Stack: `nemotron` / `nemotron-instruct` (NVFP4) vs `qwen` (dense 27B, serving variants
`qwen`, `qwen-code`, `qwen-instruct`). All think/no-think variants served via the local
LiteLLM gateway (localhost:4000 == sparky:4000). Full runs this session:

## CRUXEval-O (predict Python output, pass@1)

| model | pass | note |
|---|---|---|
| qwen-code (think, code-tuned) | 48/50 = 96% | |
| qwen (think) | 46/50 = 92% | |
| qwen-instruct (no-think) | 37/50 = 74% | |
| nemotron (think) | 41/49 = 84% | 1 request silently dropped (49/50 responded) |
| nemotron-instruct (no-think) | 20/48 = 42% | 2 requests dropped (48/50 responded) |

Leaderboard ref: GPT-4o ~74% O, Qwen2.5-72B ~71% O, Llama-3.1-70B ~64% O.

## Classifier (umpire-bot finding re-classification, 6 cases, deterministic JSON checks)

| model | pass |
|---|---|
| qwen-instruct (no-think) | 6/6 = 100% |
| qwen (think) | 5/6 = 83% |

qwen-think failed case 5 (closure-capture): correct `still_valid` classification but wrong
disposition (needs_action instead of accepted) → score 0.5. qwen-instruct nailed all 6.
(nemotron classifier light runs exist separately.)

## Bug-finding (QuixBugs, 23 programs) — execute the model's own fix

Grading = run the corrected function the model outputs against the program's full test
suite; `fixed` = passes everything the correct version passes. No LLM judge.

| model | fixed | statuses |
|---|---|---|
| qwen-instruct (no-think) | 19/23 = 83% | 19 fixed, 1 part, 3 notfixed |
| nemotron-instruct (no-think) | 15/23 = 65% | 15 fixed, 2 part, 6 notfixed |
| nemotron (think) | 15/23 = 65% | 15 fixed, 7 notfixed, 1 no-code |
| qwen (think) | 15/23 = 65% | 15 fixed, 7 notfixed, 1 no-code |

| qwen-moe (think) | 1/23 = 4% | see caveat below — collapses in think mode |
| qwen-moe-instruct (no-think) | 14/23 = 61% | 14 fixed, 2 part, 7 notfixed |

Note: qwen-moe-think's 1/23 is striking but consistent with the broader pattern — every
thinking variant either fails to help (nemotron flat) or actively hurts (qwen-dense
83%→65%). qwen-moe-think is the most extreme: it burns ~8K tokens reasoning per program
and then emits a still-buggy (or token-truncated) function. Worth re-running with a shim
if this matters to you; the no-think 14/23 is the more representative signal.

## Takeaways for reviewer/mule trialing

- **Mule = use qwen-instruct (no-think).** Fastest AND best bug-fixing (83%) — beats
  nemotron no-think (65%) by a wide margin, and beats both thinking variants too.
- **Thinking does NOT help here.** qwen dropped from 83% (no-think) to 65% (think) on
  bug-finding; nemotron was flat (65% both). If you want reasoning quality, reach for
  qwen-code (96% crux) instead of qwen (think).
- qwen is the stronger raw reasoner on crux in every serving mode (74/92/96 vs 42/84).
- Classifier: qwen no-think is a clean 6/6; qwen think is 5/6 (misfires only on an edge
  disposition case).

## Caveats

- The gateway occasionally drops a request under load (1–2 per 50-test run); percentages
  computed on responded tests. Re-running fills the gaps.
- QuixBugs grading can't reproduce a fix if the model emits prose instead of a `def`
  block (1 "no-code" per thinking model) — extract_python strips markdown, else falls
  back to the last function block.
