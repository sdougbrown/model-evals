#!/usr/bin/env python3
"""
Bug-IDENTIFICATION recall eval (QuixBugs) — decoupled from bug *correction*.

The existing quixbugs_runner.py measures correction: the model must emit a runnable
fixed function, which we execute. That conflates two jobs. A reviewer's job is mostly
IDENTIFICATION (spot the defect), not producing a clean patch. Reasoning models in
particular identify bugs but fail the correction harness (long thinking -> no clean
`def` -> fails execution). This eval fixes that attribution problem.

For each of the 23 KEEP QuixBugs programs:
  1. show buggy source + one concrete failing case (same as runner)
  2. ask the model to IDENTIFY the bug in prose (which line, why, what it should be)
  3. an LLM judge (deepseek-flash, local gateway) decides whether the finding
     matches the ground-truth single-line fix
  4. recall = fraction of programs where the model correctly identified the bug

No cloud judge. Uses deepseek-flash on the local gateway.

Usage:
  ./quixbugs_identify.py --models qwen qwen-instruct qwen-moe qwen-moe-instruct gemma4 \
      --judge deepseek-flash --gateway http://localhost:4000/v1
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quixbugs_runner import load_programs, build_prompt, extract_python  # noqa

def chat(model, messages, gateway, max_tokens=2048, timeout=900):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(gateway + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def _compose_output(msg):
    return (msg.get("content") or "") + "\n" + (msg.get("reasoning_content")
            or (msg.get("provider_specific_fields") or {}).get("reasoning") or "")

IDENTIFY_SYSTEM = (
    "You are a code reviewer examining a Python function that has exactly one subtle bug: "
    "for some inputs it returns the wrong result. Identify the defect precisely."
)
def identify_prompt(prog):
    args, exp, got = prog["failing"]
    args_lit = ", ".join(repr(a) for a in args)
    return f"""The following Python function has a subtle bug — it produces a wrong result for some inputs.

```python
{prog['buggy']}
```

Concrete failing case:

f({args_lit})
  expected: {exp}
  actual:   {got}

Identify the bug. In your reply (prose, no code-block requirement), state:
1. the EXACT line/expression that is wrong (quote it),
2. WHY it is wrong,
3. what the correct expression/line should be.

Be precise and concise. You do NOT need to output a full corrected function — only identify the defect."""

JUDGE_SYSTEM = (
    "You are a strict code-review grader. A reviewer was shown a buggy Python function "
    "and asked to identify the bug. You will receive: the BUGGY source, the CORRECT source "
    "(differing by the single intended fix), and the REVIEWER'S finding. Decide whether the "
    "finding correctly identifies the actual root-cause bug that the fix addresses. "
    "Output ONLY JSON: {\"correct\": true/false, \"reason\": \"short justification\"}. "
    "true means the finding pinpoints the same defect (matching the changed line/expression), "
    "even if phrased differently. false means the finding names a different/nonexistent issue, "
    "is too vague to pin the real bug, or is wrong."
)

def judge_prompt(prog):
    return f"""<buggy_source>
{prog['buggy']}
</buggy_source>

<correct_source>
{prog['correct']}
</correct_source>

<reviewer_finding>
{prog['finding']}
</reviewer_finding>

Please output your JSON verdict exactly."""

def grade_identification(model, programs, gateway, judge, max_samples=None):
    results = []
    for i, prog in enumerate(programs):
        if max_samples and i >= max_samples:
            break
        name = prog["name"]
        if prog["failing"] is None:
            continue
        try:
            resp = chat(model, [{"role": "user", "content": identify_prompt(prog)}], gateway, max_tokens=1500)
            finding = _compose_output(resp["choices"][0]["message"])
        except Exception as e:
            results.append({"prog": name, "model_status": "gateway_error", "detail": str(e)})
            continue
        if not finding.strip():
            results.append({"prog": name, "model_status": "no_output"})
            continue
        prog["finding"] = finding
        try:
            jres = chat(judge, [{"role": "system", "content": JUDGE_SYSTEM},
                                {"role": "user", "content": judge_prompt(prog)}], gateway, max_tokens=800)
            jtxt = _compose_output(jres["choices"][0]["message"]).strip()
        except Exception as e:
            results.append({"prog": name, "model_status": "judge_error", "detail": str(e)})
            continue
        correct = None
        try:
            m = re.search(r"\{.*\}", jtxt, re.S)
            correct = json.loads(m.group(0)).get("correct")
        except Exception:
            correct = None
        results.append({"prog": name, "finding": finding[:600], "judge": jtxt[:300],
                        "correct": correct, "kb": "correct" if correct else ("ambig" if correct is None else "wrong")})
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--judge", default="deepseek-flash")
    ap.add_argument("--gateway", default="http://localhost:4000/v1")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--save", default="results/quixbugs-identify.json")
    args = ap.parse_args()

    programs = load_programs()
    print(f"Loaded {len(programs)} KEEP programs for identification")
    allout = {}
    for model in args.models:
        st = time.time()
        res = grade_identification(model, programs, args.gateway, args.judge, args.max_samples)
        rc = sum(1 for r in res if r.get("correct") is True)
        n = len(res)
        wrong = sum(1 for r in res if r.get("correct") is False)
        ambig = sum(1 for r in res if r.get("correct") is None)
        print(f"\n== {model} ==\n  recall (correct/total) = {rc}/{n}   wrong={wrong}  ambiguous/judge-err={ambig}")
        allout[model] = {"results": res, "recall": rc, "n": n, "seconds": round(time.time()-st)}
        if args.save:
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            with open(args.save, "w") as f:
                json.dump(allout, f, indent=2)
    print("\n=== RECALL SUMMARY ===")
    for model in allout:
        a = allout[model]
        print(f"  {model:26s} {a['recall']}/{a['n']}")

if __name__ == "__main__":
    main()
