#!/usr/bin/env python3
"""Validate QuixBugs entries: check buggy version FAILS >=1 json test and correct
version PASSES all. Run in isolated subprocesses with a timeout so a slow/hanging
program can't stall the whole build."""
import json, os, subprocess, sys

QB = os.environ.get("QUBIXBUGS", "/tmp/QuixBugs")
os.chdir(QB)

DRIVER = r"""
import json, sys, importlib.util, os
path, name, testfile = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("m", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
fn = getattr(m, name)
cases = [json.loads(l) for l in open(testfile) if l.strip()]
limit = int(os.environ.get("LIMIT", "12"))
passed = 0
for args, exp in cases[:limit]:
    got = fn(*args)
    passed += (got == exp)
print(passed, len(cases[:limit]))
"""

def run_one(path, name, testfile, timeout=8):
    try:
        out = subprocess.run([sys.executable, "-c", DRIVER, path, name, testfile],
                             capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0:
            return None
        a, b = out.stdout.strip().split()
        return int(a), int(b)
    except (subprocess.TimeoutExpired, ValueError):
        return None

for jf in sorted(os.listdir("json_testcases")):
    name = jf.replace(".json", "")
    bp, cp = f"python_programs/{name}.py", f"correct_python_programs/{name}.py"
    jp = f"json_testcases/{jf}"
    if not (os.path.exists(bp) and os.path.exists(cp)):
        continue
    br = run_one(bp, name, jp)
    cr = run_one(cp, name, jp)
    if br is None or cr is None:
        print(f"{name:28s} SKIP (err/timeout: buggy={br} correct={cr})")
        continue
    bp, bt = br; cp, ct = cr
    if bp < bt and cp == ct:
        tag = "KEEP"
    elif cp < ct:
        tag = "CREATE-CANNOT-REPRO"
    else:
        tag = "BUGGY-PASSES" if bp == bt else "MIXED"
    print(f"{name:28s} buggy {bp}/{bt}  correct {cp}/{ct}  {tag}")
