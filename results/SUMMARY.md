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

Note: qwen-moe-think's 1/23 was largely a token-exhaustion artifact, NOT a true ability
ceiling. A 24K-token re-run on the 3 programs it had collapsed on fixed 2/3 (bucketsort,
gcd fixed; find_in_sorted still broken). Longer leash recovers most of its bug-fixing.
Even so, thinking is not a free win: find_in_sorted stayed notfixed, and no-think
qwen-moe-mult is 14/23 while qwen-instruct no-think is 19/23. Full long-leash run is the
remaining open question (~1-1.5h for all 23 at 24K tokens).

## Bug-IDENTIFICATION recall (QuixBugs, judge=deepseek-flash)

Decouples spotting the bug from producing a runnable fix. Each model identifies the defect
in prose; deepseek-flash (local gateway) judges whether the finding matches the single-line
fix. Full matrix: results/IDENTIFY_MATRIX.txt

| model | recall | vs its own correction |
|---|---|---|
| gemma4 | 19/23 = 83% | (not run on correction) |
| qwen (think) | 18/23 = 78% | corr 15/23 |
| qwen-code (think) | 16/23 = 70% | (not run on correction) |
| qwen-instruct (no-think) | 15/23 = 65% | corr 19/23 |
| qwen-moe-instruct (no-think) | 14/23 = 61% | corr 14/23 |
| qwen-moe (think) | 13/23 = 57% | corr 6/23 (24K) |
| nemotron (think) | 10/23 = 43% | corr 15/23 |
| nemotron-instruct (no-think) | 10/23 = 43% | corr 15/23 |
| gemma4 reason=high | 20/23 = 87% | best identifier recorded |
| gemma4 reason=low | 19/23 = 83% | = no-think |
| gemma4 reason=medium | 16/23 = 70% | hurts recall |

Key finding: the correction harness systematically under-credits thinking models. qwen-moe
identifies 13/23 but appears to fix only 6/23; qwen identifies 18/23 (fixes 15). A reviewer
is judged on IDENTIFICATION recall, so use this number, not the correction number.
Run: python3 scripts/quixbugs_identify.py --models <...> --judge deepseek-flash

Gemma4 reasoning sweep (gateway `reasoning` param): high=20/23 best anywhere; low=no-think
(19/23); medium=16/23 (actively worse). Supplemental probe — if gemma tooling is made
stable, reasoning=high gives it the top reviewer recall.

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

## qwen FP rate vs thinking budget
FP on 23 bug-free programs (lower=better). Thinking budget trades precision for recall:
| reasoning | (=old model) | FP | identify recall |
|---|---|---|---|
| none | qwen-instruct | 1/23 = 4% | 15/23 |
| low | qwen-code | 2/23 = 9% | 16/23 |
| max (default) | qwen | 3/23 = 13% | 18/23 |
| high | capped-high | 4/23 = 17% | (pending) |
| xhigh | capped-xhigh | 6/23 = 26% | (pending) |
Monotonic: more thinking => more false alarms on clean code (4% -> 26%). high/xhigh recall (identify)
not yet measured; would complete the tradeoff curve.

## Judge / triage-verification eval
Each model VERIFIES the same 41 findings (gemma-no-think's identifications on buggy code +
phantom findings on clean code), reference = deepseek-flash verdict. Okay to reuse.
| judge | agree (41) | posts-bad | suppresses-real |
|---|---|---|---|
| deepseek-flash | 29 = 71% | 8 | 4 |
| qwen (low) | 23 = 56% | 14 | 3 |
| qwen (max) | 23 = 56% | 12 | 3 |
| nemotron-instruct | 22 = 54% | 9 | 10 |
| gemma4 | 20 = 49% | 19 | 2 |
| gemma4-high | 20 = 49% | 19 | 2 |
| nemotron | 18 = 44% | 15 | 2 |
posts-bad = would post a phantom; suppresses-real = would drop a real finding. qwen is the best
local judge and the only strong model on both discovery and verification; gemma discovers well
but rubber-stamps as judge; nemotron-instruct is over-conservative (drops real findings).
