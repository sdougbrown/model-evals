#!/usr/bin/env python3
"""
Judge/triage eval — how well does a local model VERIFY findings, vs the reference
judge (deepseek-flash, a larger model we treat as ground truth)?

Mirrors umpire-bot triage: given code + a reviewer's finding, decide whether the
finding is correct before posting.

Dataset: a mixed set of findings that CLAIM a bug, with reference labels:
  * gemma4 no-think identifications on 23 BUGGY programs where the finding is
    correct (deepseek correct=True) -> reference "accept"
  * ... where it is WRONG (correct=False)            -> reference "reject"
  * gemma4 no-think findings on CLEAN (correct) code where it falsely claimed
    a bug (FP true)                                  -> reference "reject"
Each candidate judge model sees (code-under-review, finding) and outputs a
correct/incorrect verdict. We score agreement with the reference, plus the
triage-critical confusion matrix (false accept = would post a bad finding;
false reject = would suppress a real finding).

Usage:
  ./quixbugs_judge.py --judges qwen qwen-low gemma4 nemotron ...
"""
import argparse, json, os, re, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quixbugs_runner import load_programs

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

def chat(model, messages, gateway, max_tokens=800, timeout=600, reasoning=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if reasoning:
        body["reasoning"] = reasoning
    req = urllib.request.Request(gateway + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def _compose(msg):
    return (msg.get("content") or "") + "\n" + (msg.get("reasoning_content")
            or (msg.get("provider_specific_fields") or {}).get("reasoning") or "")

JUDGE_SYSTEM = (
    "You are a strict code-review verifier. You are given a piece of code and a "
    "reviewer's finding. Decide whether the finding CORRECTLY identifies a real bug. "
    "Think carefully, but your final message must be EXACTLY one JSON object, nothing "
    "before or after it: {\"correct\": true/false, \"reason\": \"brief\"}. "
    "correct=true only if the finding pinpoints an actual defect in the given code. "
    "A finding that names a nonexistent issue, or that is a false alarm on otherwise "
    "valid code, is correct=false."
)

def build_task(code_src, finding, reference):
    return {"src": code_src, "finding": finding, "reference": reference}

def judge_prompt(task):
    return f"""<code_under_review>
{task['src']}
</code_under_review>

<reviewer_finding>
{task['finding']}
</reviewer_finding>

Output your JSON verdict."""

def build_dataset():
    progs = {p["name"]: p for p in load_programs()}
    tasks = []
    # 1) buggy-program identifications from gemma4 no-think (have 'correct' reference)
    ident = json.load(open(os.path.join(RESULTS, "quixbugs-identify.json")))["gemma4"]["results"]
    used_buggy = 0
    for r in ident:
        if "correct" not in r:
            continue
        p = progs[r["prog"]]
        reference = bool(r["correct"])
        tasks.append({
            "id": f"buggy-{r['prog']}",
            "prog": r["prog"], "src": p["buggy"],
            "finding": r["finding"], "reference": reference,
            "kind": "buggy", "ref_label": ("accept" if reference else "reject"),
        })
    # 2) clean-code false-positive findings from gemma4 no-think (reference=reject)
    fp = json.load(open(os.path.join(RESULTS, "quixbugs-falsepos-gemma-nothink.json")))["gemma4"]["results"]
    for r in fp:
        claimed = r.get("claimed")
        if claimed is not True:
            continue  # only findings where the reviewer asserted a bug
        p = progs[r["prog"]]
        tasks.append({
            "id": f"clean-{r['prog']}", "prog": r["prog"], "src": p["correct"],
            "finding": r["finding"], "reference": False, "kind": "clean",
            "ref_label": "reject",
        })
    return tasks

def _parse_verdict(j):
    try:
        m = re.search(r"\{.*\}", j, re.S)
        if m:
            return json.loads(m.group(0)).get("correct")
    except Exception:
        pass
    # lenient fallback: find “correct” followed by true/false, or a lone true/false
    m = re.search(r"(?i)correct[\"'\s:=]*(true|false)", j)
    if m:
        return m.group(1).lower() == "true"
    m = re.search(r"\b(true|false)\b", j, re.I)
    if m:
        return m.group(1).lower() == "true"
    return None


def run_judge(model, tasks, gateway, reasoning=None, save=None):
    out = []
    for t in tasks:
        try:
            j = _compose(chat(model, [{"role": "system", "content": JUDGE_SYSTEM},
                                      {"role": "user", "content": judge_prompt(t)}],
                              gateway, max_tokens=1200, reasoning=reasoning)["choices"][0]["message"]).strip()
        except Exception as e:
            out.append({"id": t["id"], "status": "gateway_error", "detail": str(e)})
            continue
        verdict = _parse_verdict(j)
        agree = None if verdict is None else (verdict == t["reference"])
        out.append({"id": t["id"], "prog": t["prog"], "kind": t["kind"],
                    "verdict": verdict, "reference": t["reference"],
                    "agree": agree, "judge": j[:300], "raw": j,
                    "fa": verdict is not None and verdict is True and t["reference"] is False,
                    "fr": verdict is not None and verdict is False and t["reference"] is True})
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        json.dump({model: out}, open(save, "w"), indent=2)
    return out

def summarize(model, out):
    j = [o for o in out if "agree" in o]
    n = len(j)
    agree = sum(1 for o in j if o["agree"])
    fa = sum(1 for o in j if o.get("fa"))
    fr = sum(1 for o in j if o.get("fr"))
    print(f"  {model:22s} agree {agree}/{n} ({round(100*agree/n)}%)   posts-bad {fa}   suppresses-real {fr}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", required=True)
    ap.add_argument("--reasoning", choices=["low","medium","high","none","max","xhigh"], default=None)
    ap.add_argument("--gateway", default="http://localhost:4000/v1")
    ap.add_argument("--save", default="results/quixbugs-judge.json")
    args = ap.parse_args()

    tasks = build_dataset()
    pos = sum(1 for t in tasks if t["reference"])
    print(f"Judge dataset: {len(tasks)} findings  (accept={pos}, reject={len(tasks)-pos})")
    allout = {}
    for model in args.judges:
        out = run_judge(model, tasks, args.gateway, args.reasoning, args.save)
        summarize(model, out)
        allout[model] = out
    if args.save:
        json.dump(allout, open(args.save, "w"), indent=2)
    print("\ntriage angles: posts-bad = false accept (bad finding would be posted); suppresses-real = false reject.")

if __name__ == "__main__":
    main()
