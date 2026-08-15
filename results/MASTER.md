# Master matrix — every family × every eval (2026-08-14 session)

All current-session numbers, local gateway. Full detail in SUMMARY.md + raw JSON.

| model | crux pass@1 (50) | classifier (6) | identify recall (23) | FP rate (23) |
|---|---|---|---|
| qwen-code (think, code-tuned) | 48 = 96% | 5 | 16 | n/a * |
| qwen (think) | 46 = 92% | 5 | 18 | 13% |
| qwen-instruct (no-think) | 37 = 74% | 6 | 15 | n/a * |
| nemotron (think) | 41 = 84% | 5 | 10 | 22% |
| nemotron-instruct (no-think) | 20 = 42% | 3 | 10 | 30% |
| gemma4 (no-think) | 35 = 70% (retuned) | 6 | 21 | 48% |
| gemma4 (reasoning=high) | 35 = 70%* (retuned) | 6 | 20 | 57% |
| qwen-moe (think) | 30 = 60% (hist) | 5 | 13 | (dropped) |
| qwen-moe-instruct (no-think) | 27 = 54% (hist) | 3 | 14 | (dropped) |

*gemma4 reasoning=high crux: 33/50 clean passes + 17 gateway errors (model is flaky/slow
under reasoning-high; masked by the assertion format). Same number as no-think either way.

## Notes / gotchas
- promptfoo dedupes providers sharing the same `id` across labels — to compare two serving
  configs of one model, run them as SEPARATE configs (as done here), or the "reasoning" row
  will silently reuse the no-reasoning output.
- reasoning=high changes gemma's output format (trailing reasoning), so crux's "take last
  line" assertion fails on it regardless of correctness. Identify/classifier don't have that
  failure mode, which is why gemma-high looks fine there but impossible to extract on crux.
- gateway occasionally drops requests under concurrent load (1–2 per 50-test run, more with
  gemma reasoning-high).

## Decision aids
- Reviewer (identification recall): gemma4(no-think, retuned) 21 > gemma4(high) 20 > qwen 18
  > qwen-code 16 > qwen-instruct 15 > qwen-moe 13-14 > nemotron 10. Retuned sampling
  lifted gemma no-think recall 19 -> 21; high unchanged at 20.
- False-positive rate (bug-free code, lower=better): qwen 13% < nemotron 22% <
  nemotron-instruct 30% < gemma4 48% < gemma4-high 57%. qwen is the best balanced
  reviewer (high recall + low FP); gemma has the best recall but cries wolf (~half of
  clean code flagged), worse with reasoning=high. FP runs: scripts/quixbugs_falsepos.py
  on the CORRECT (bug-free) sources; qwen-instruct/qwen-code n/a (not served when run).
- Mule (correction / cheap fast output): qwen-instruct.
- Reasoning-heavy crux: qwen-code > qwen > nemotron.
- nemotron is the weakest identifier of the lineup; no-think nemotron/qwen-moe are also weak
  on classifier (3/6).
