#!/usr/bin/env python3
"""
Bug false-positive (FP) eval (QuixBugs) — how often does a model flag a bug
that isn't there?

Uses each program's CORRECT (bug-free) source. We show the model clean code, ask it
to report any bugs (or say NONE), then a judge (deepseek-flash) decides whether any
claimed issue is real. Since the code is bug-free, ANY claimed bug = false positive.

Metrics per model:
  FP rate = #programs where the model wrongly claimed a bug / #programs
  (complement is "correctly said no bug" recall of absence)

Usage:
  ./quixbugs_falsepos.py --models gemma4 qwen ... --gateway http://localhost:4000/v1
"""
import argparse, json, os, re, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quixbugs_runner import load_programs  # noqa

def chat(model, messages, gateway, max_tokens=1500, timeout=600, reasoning=None, extra_body=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if reasoning:
        body["reasoning"] = reasoning
    if extra_body:
        body["extra_body"] = extra_body
    req = urllib.request.Request(gateway + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def _compose(msg):
    return (msg.get("content") or "") + "\n" + (msg.get("reasoning_content")
            or (msg.get("provider_specific_fields") or {}).get("reasoning") or "")

def review_prompt(correct_src):
    return f"""Review the following Python function for bugs.

```python
{correct_src}
```

If the function has NO bug, reply with exactly the single token: NO ISSUE
If it DOES have a bug, describe each one precisely: the exact line/expression,
why it's wrong, and what it should be. Only report bugs you are confident exist."""

JUDGE_SYSTEM = (
    "You are a strict code reviewer. A model was shown a Python function and asked to "
    "report any bugs, or say 'NO ISSUE'. You will be given the function source and the "
    "model's reply. Determine whether the reply wrongly claims a bug exists. "
    "The function may or may not actually have a bug. Output ONLY JSON: "
    "{\"claimed_bug\": true/false, \"real_bug\": true/false, \"reason\": \"...\"}. "
    "claimed_bug=true if the model asserted there is a bug. real_bug=true if the "
    "function genuinely contains the bug the model described. A false positive is "
    "claimed_bug=true AND real_bug=false. If the model says NO ISSUE, claimed_bug=false."
)

def judge_prompt(correct_src, finding):
    return f"""<function_source>
{correct_src}
</function_source>

<model_reply>
{finding}
</model_reply>

Output your JSON verdict."""

def eval_model(model, programs, gateway, judge, reasoning=None, save=None, label=None):
    results = []
    for i, prog in enumerate(programs):
        name = prog["name"]
        src = prog["correct"]
        try:
            resp = chat(model, [{"role": "user", "content": review_prompt(src)}], gateway,
                        max_tokens=1500, reasoning=reasoning)
            finding = _compose(resp["choices"][0]["message"]).strip()
        except Exception as e:
            results.append({"prog": name, "status": "gateway_error", "detail": str(e)})
            continue
        if not finding:
            results.append({"prog": name, "status": "no_output"})
            continue
        try:
            j = _compose(chat(judge, [{"role": "system", "content": JUDGE_SYSTEM},
                                      {"role": "user", "content": judge_prompt(src, finding)}],
                              gateway, max_tokens=600)["choices"][0]["message"]).strip()
        except Exception as e:
            results.append({"prog": name, "status": "judge_error", "detail": str(e)})
            continue
        try:
            jj = json.loads(re.search(r"\{.*\}", j, re.S).group(0))
            claimed = jj.get("claimed_bug"); real = jj.get("real_bug")
        except Exception:
            claimed = real = None
        fp = (claimed is True) and (real is not True)
        results.append({"prog": name, "claimed": claimed, "real": real, "fp": fp,
                        "finding": finding[:500], "judge": j[:200]})
    n = sum(1 for r in results if r.get("status") in (None,)) or len(results)
    # count only judged rows
    judged = [r for r in results if "fp" in r]
    fp_count = sum(1 for r in judged if r["fp"])
    return results, fp_count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--reasoning", choices=["low","medium","high","none","max","xhigh","minimal"], default=None)
    ap.add_argument("--judge", default="deepseek-flash")
    ap.add_argument("--gateway", default="http://localhost:4000/v1")
    ap.add_argument("--save", default="results/quixbugs-falsepos.json")
    args = ap.parse_args()

    programs = load_programs()
    print(f"Loaded {len(programs)} KEEP programs (using CORRECT/bug-free sources)")
    key = lambda m: (m + (("-" + args.reasoning) if args.reasoning else ""))
    allout = {}
    for model in args.models:
        results, fp = eval_model(model, programs, args.gateway, args.judge, args.reasoning, args.save)
        judged = [r for r in results if r.get("fp") is not None]
        fp_rate = fp / len(judged) if judged else None
        print(f"\n== {key(model)} ==  false positives: {fp}/{len(judged)}  ({round(100*fp_rate) if fp_rate else 0}%)")
        allout[key(model)] = {"results": results, "fp": fp, "n": len(judged),
                              "noissue_clean": len(judged) - fp}
        if args.save:
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            json.dump(allout, open(args.save, "w"), indent=2)
    print("\n=== FP RATE SUMMARY (lower = better) ===")
    for m in allout:
        a = allout[m]
        pct = round(100 * a["fp"] / a["n"]) if a["n"] else 0
        print(f"  {m:26s} {a['fp']}/{a['n']} = {pct}% falsely flagged")

if __name__ == "__main__":
    main()
