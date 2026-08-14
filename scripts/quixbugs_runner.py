#!/usr/bin/env python3
"""
Bug-finding eval (QuixBugs Python) — deterministic, self-contained.

For each QuixBugs entry:
  - present the BUGGY program + one failing test case (args → expected vs actual)
  - ask the model to output the complete corrected function
  - run the model's function against the FULL set of that program's json test cases
  - "fixed"  => model's function passes all tests that the correct version passes
  - "part"   => model's function passes some (bug partially addressed)
  - "notfixed"/"noexecute" otherwise

Deterministic grading: we execute the model's own corrected code against real
known inputs. No LLM-as-judge, no cloud API needed (only the local gateway).

Usage:
  ./quixbugs_runner.py --models qwen qwen-instruct nemotron --gateway http://localhost:4000/v1
"""
import argparse, importlib, importlib.util, json, os, re, subprocess, sys, tempfile, time

QB = os.environ.get("QUBIXBUGS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evals", "bugfinding", "quixbugs"))

DRIVER = r"""
import json, sys, importlib.util, os
path, name, testfile = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("m", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
fn = getattr(m, name)
cases = [json.loads(l) for l in open(testfile) if l.strip()]
limit = int(os.environ.get("LIMIT", "9999"))
passed = 0
for args, exp in cases[:limit]:
    try:
        got = fn(*args)
    except Exception:
        continue
    passed += (got == exp)
print(passed, len(cases[:limit]))
"""

def _run_code(path, name, testfile, timeout=10):
    try:
        out = subprocess.run([sys.executable, "-c", DRIVER, path, name, testfile],
                             capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0:
            return None
        a, b = out.stdout.strip().split()
        return int(a), int(b)
    except (subprocess.TimeoutExpired, ValueError):
        return None

def load_programs():
    """Return list of (name, buggy_source, correct_source, testfile, failing_case).
    failing_case = first test case the buggy version fails, as (args, expected, actual).
    Only KEEP entries (buggy fails >=1, correct passes all)."""
    out = []
    for jf in sorted(os.listdir(os.path.join(QB, "json_testcases"))):
        name = jf.replace(".json", "")
        bp = os.path.join(QB, "python_programs", name + ".py")
        cp = os.path.join(QB, "correct_python_programs", name + ".py")
        jp = os.path.join(QB, "json_testcases", jf)
        if not (os.path.exists(bp) and os.path.exists(cp)):
            continue
        with open(jp) as f:
            cases = [json.loads(l) for l in f if l.strip()]
        br = _run_code(bp, name, jp)
        cr = _run_code(cp, name, jp)
        if not br or not cr:
            continue
        bp_pass, bt = br; cp_pass, ct = cr
        if not (bp_pass < bt and cp_pass == ct):
            continue
        # find first failing case for the hint
        mod_b = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        mod_b.write(open(bp).read()); mod_b.close()
        failing = None
        m = _load(bp, name)
        failing_candidates = []
        for args, exp in cases:
            try:
                got = m[name](*args)
            except Exception:
                got = "<exception>"
            if got != exp:
                res_len = len(repr(args)) + len(repr(exp)) + len(repr(got))
                failing_candidates.append((res_len, (args, exp, got)))
        if failing_candidates:
            # prefer the most compact failing case so the prompt hint stays small
            failing_candidates.sort(key=lambda x: x[0])
            failing = failing_candidates[0][1]
        with open(bp) as f:
            bs = f.read()
        with open(cp) as f:
            cs = f.read()
        out.append({
            "name": name, "buggy": bs, "correct": cs, "testfile": jp,
            "cases": cases, "failing": failing,
            "buggy_pass": bp_pass, "total": bt,
            "correct_pass": cp_pass,
        })
    return out

def _load(path, name):
    spec = importlib.util.spec_from_file_location("q_" + name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def gateway_chat(model, messages, gateway, max_tokens=8192):
    import urllib.request
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(gateway + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)

def build_prompt(prog):
    name = prog["name"]
    args, exp, got = prog["failing"]
    args_lit = ", ".join(repr(a) for a in args)
    return f"""You are a bug-finding agent. The following Python function has a subtle bug: it produces
incorrect output for some inputs.

```python
{prog['buggy']}
```

Here is a concrete failing case:

f({args_lit})
  expected: {exp}
  actual:   {got}

Find the bug and fix it. Output ONLY the complete corrected function as a single
Markdown python code block (```python ... ```). Do not include explanations,
tests, or extra code — only the one corrected function definition with the same
function name and signature.
"""

def extract_python(text):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S | re.I)
    if blocks:
        # prefer a block containing "def "
        for b in blocks:
            if "def " in b:
                return b
        return blocks[0]
    # fallback: try to find a def ... function in raw text
    m = re.search(r"(def\s+\w+\(.*?)(?=\n\s*\ndef\s|\Z)", text, re.S)
    if m:
        return m.group(1)
    return text


def grade_model(model, programs, gateway, max_samples=None, sleep=0.0, max_tokens=8192):
    results = []
    for i, prog in enumerate(programs):
        if max_samples and i >= max_samples:
            break
        name = prog["name"]
        if prog["failing"] is None:
            continue
        msgs = [{"role": "user", "content": build_prompt(prog)}]
        try:
            resp = gateway_chat(model, msgs, gateway, max_tokens=max_tokens)
        except Exception as e:
            results.append({"prog": name, "status": "gateway_error", "detail": str(e)})
            continue
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or (msg.get("provider_specific_fields") or {}).get("reasoning") or ""
        combined = content + "\n" + reasoning
        code = extract_python(combined)
        # write candidate module and test it
        fd, tmp = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, "w") as f:
            f.write(code)
        try:
            got = _run_code(tmp, name, prog["testfile"])
        except Exception:
            got = None
        finally:
            os.unlink(tmp)
        if got is None:
            status = "noexecute"
        else:
            p, t = got
            status = "fixed" if p == prog["correct_pass"] else ("part" if p > prog["buggy_pass"] else "notfixed")
        results.append({"prog": name, "status": status, "passed": got and got[0],
                        "need": prog["correct_pass"], "sample": content[:300]})
        if sleep:
            time.sleep(sleep)
    return results

def summarize(model, results):
    n = len(results)
    fixed = sum(1 for r in results if r["status"] == "fixed")
    part = sum(1 for r in results if r["status"] == "part")
    notfixed = sum(1 for r in results if r["status"] == "notfixed")
    noexec = sum(1 for r in results if r["status"] in ("noexecute", "gateway_error"))
    total_pass = fixed + part
    print(f"\n=== {model} ===")
    print(f"  fixed: {fixed}/{n}   part: {part}   notfixed: {notfixed}   noexec/no-code: {noexec}")
    return {"model": model, "fixed": fixed, "part": part, "n": n}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["nemotron"])
    ap.add_argument("--gateway", default="http://localhost:4000/v1")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--save", default="results/quixbugs.json")
    args = ap.parse_args()

    programs = load_programs()
    n_keep = len(programs)
    print(f"Loaded {n_keep} KEEP QuixBugs programs:")
    for p in programs:
        fail = p["buggy_pass"]
        print(f"  {p['name']:26s} buggy passes {p['buggy_pass']}/{p['total']}  (first failing: f({p['failing'][0]}) expected {p['failing'][1]!r})")

    allout = {}
    for model in args.models:
        st = time.time()
        res = grade_model(model, programs, args.gateway, args.max_samples, max_tokens=args.max_tokens)
        summ = summarize(model, res)
        allout[model] = {"results": res, "summary": summ, "seconds": round(time.time() - st)}
        if args.save:
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            with open(args.save, "w") as f:
                json.dump(allout, f, indent=2)

    print("\n=== SUMMARY ===")
    for model in allout:
        s = allout[model]["summary"]
        print(f"  {model}: {s['fixed']} fixed / {s['n']} kept  (part {s['part']})")

if __name__ == "__main__":
    main()
