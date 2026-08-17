# qwen-3.8:27b (replaces qwen-3.6 27b) — eval results across native thinking modes

Gateway model `qwen-3.8:27b` on sparky:4000. Native thinking modes: minimal (= low thinking
+ qwen-code-style sampling), low, medium, (high/xhigh unbounded). Compared vs prior lineup.

| eval | minimal | low | medium | reference (best elsewhere) |
|---|---|---|---|---|
| CRUX pass@1 (50) | 48/49 = 98% | 47/49 = 96% | 47/48 = 98% | qwen-3.6 code 96% |
| identify recall (23, 4096-tok) | 15/23 = 65% | 15/23 = 65% | 18/23 = 78% | gemma4 21/23 (but 48% FP) |
| false-positive rate (23) | 2/23 = 9% | 4/23 = 17% | 4/23 = 17% | qwen-3.6 (min) 13% |
| judge agree (41) | 21/41 = 51% | 24/41 = 59% | 25/41 = 61% | deepseek 71% |

Notes:
- CRUX is elite across ALL modes (96-98%); qwen-3.8 is the strongest code-reasoner measured
  in this repo to date. Mode barely matters (even "minimal" hits 98%).
- identify recall (4096-token re-run): minimal/low 65%, medium 78% — medium matches
  qwen-3.6's best (18/23) but with far lower FP (17% vs gemma's 48%). Mode raises recall
  (65->78%) but also FP (9->17%).
- judge: medium (61%) is the best LOCAL judge seen (beats qwen-3.6's 56%; deepseek 71% is
  the ceiling). posts-bad=11, suppresses-real=2-3 across modes.
- false positives rise with thinking budget (9% -> 17%), same direction as qwen-3.6.
