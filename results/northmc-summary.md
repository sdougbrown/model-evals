# north-mini-code — QuixBugs-family evals (low / medium thinking)
| eval | low | medium | ref: qwen-3.8 |
|---|---|---|---|
| CRUX (50) | 0%* | 0%* | 98% |
| identify recall (23) | 12 = 52% | 15 = 65% | 65 / 78% |
| false-positive (23) | 6 = 26% | 7 = 30% | 9 / 17% |
| judge (41) | 44% | 41% | 51 / 61% |

*CRUX 0% is a FORMAT artifact, not ability: north-mini-code emits its whole CoT inline in
content (`<|START_THINKING|>...`) and, under our crux "reply with only the literal" prompt
+ last-line assertion, never lands a clean literal within budget. Its crux is not gradable
with the current prompt/assertion. To get a true crux number we'd need a north-aware prompt
(force "final line = the literal") and/or a normalizer that strips the inline thinking block.
Note the user recalls north scoring low-90s on crux historically with a different setup.

Reads (valid py evals):
- identify recall low 52% / medium 65% — below qwen-3.8 (65/78) and gemma (21 =91%).
- false-positive 26-30% — on the high side (fewer than gemma's 48%, more than qwen-3.8's 17%).
- judge 41-44% — weak verifier (qwen-3.8 51-61%, deepseek 71%).
