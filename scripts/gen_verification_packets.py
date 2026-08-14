#!/usr/bin/env python3
"""
Generate per-finding verification packets for label QA.

For each harvested finding, emit a compact JSON packet containing everything a
verifier needs to judge classification + disposition against real code:
  - finding claim (body)
  - metadata (repo, pr, path, line, head_sha)
  - prior_finding_id
  - current label (classification + disposition) and dispositions_evidence
  - code region at HEAD_sha (the state the reviewer checked) around the finding line
  - code region at latest HEAD (to confirm fixes) around the finding line
  - the stored discussion (sentiment/reactions) if any

Output: one JSON object per line -> verification_packets.jsonl
"""
import json, subprocess, os, sys, argparse

clones = {
  "sdougbrown/avenor": "/home/douglasbrown/Code/avenor",
  "sdougbrown/umpire-bot": "/home/douglasbrown/Code/umpire-bot",
  "sdougbrown/writetighter": "/home/douglasbrown/Code/writetighter",
  "sdougbrown/daywatch-cal": "/home/douglasbrown/Code/daywatch-cal",
  "umpire-tools/umpire-go-gen": "/var/lib/pr-reviewer/repos/umpire-tools/umpire-go-gen.git",
}
CONTEXT = 10

def git_show(clone, rev, path):
    r = subprocess.run(["git","-C",clone,"show",rev+":"+path],capture_output=True,text=True)
    return r.stdout if r.returncode==0 else None

def region(text, line, context=CONTEXT):
    if text is None:
        return None
    lines = text.splitlines()
    if not lines:
        return None
    n = len(lines)
    lo = max(0, (line or 1)-context-1)
    hi = min(n, (line or 1)+context)
    return "\n".join("%5d| %s" % (i+1, lines[i]) for i in range(lo, hi))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="classifications.jsonl")
    ap.add_argument("--out", default="verification_packets.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    n_written = 0
    n_missing_code = 0
    with open(args.out, "w") as out:
        for f in rows:
            repo = f["metadata"]["repo"]
            clone = clones.get(repo)
            sha = f["metadata"]["head_sha"]
            path = f["metadata"]["path"]
            line = f["metadata"]["line"]
            loc = None
            if f.get("current_locations"):
                loc = f["current_locations"][0]
            verify_path = (loc or {}).get("path") or path
            verify_line = (loc or {}).get("line") or line

            code_at_sha = git_show(clone, sha, verify_path) if clone else None
            code_at_head = git_show(clone, "HEAD", verify_path) if clone else None
            if code_at_sha is None:
                n_missing_code += 1

            packet = {
                "metadata": {
                    "repo": repo,
                    "pr_number": f["metadata"]["pr_number"],
                    "comment_id": f["metadata"]["comment_id"],
                    "prior_finding_id": f["metadata"]["prior_finding_id"],
                    "head_sha": sha,
                    "finding_path": path,
                    "finding_line": line,
                    "verify_path": verify_path,
                    "verify_line": verify_line,
                },
                "current_label": {"classification": f["classification"], "disposition": f["disposition"]},
                "claim": f["finding_body"],
                "evidence": f.get("evidence"),
                "disposition_evidence": f.get("disposition_evidence"),
                "discussion": f.get("discussion"),
                "code_at_head_sha": region(code_at_sha, verify_line),
                "code_at_latest_HEAD": region(code_at_head, verify_line),
            }
            out.write(json.dumps(packet) + "\n")
            n_written += 1
    print("wrote", n_written, "packets to", os.path.abspath(args.out))
    print("findings with NO reachable code at head_sha:", n_missing_code)

if __name__ == "__main__":
    main()
