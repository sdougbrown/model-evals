#!/usr/bin/env python3
"""
Extract hard-negative training examples for the code classifier.

Hard negative = a finding the model must learn NOT to push as needs_action, even though
it is live. These are the calibration cases where precision is won:

  Type 'accepted'      : classification still_valid (code issue genuinely exists) BUT
                         disposition accepted (developer explicitly accepted/dismissed).
                         This is the highest-value hard negative -- it separates "the code
                         pattern is real" from "this needs a fix now".
  Type 'informational' : live but intentionally low-action (per system prompt rules).
  Type 'needs_review'  : low-confidence / ambiguous -- the model should defer to a human,
                         NOT auto-repost.

These mirror the spike's hard-negative collection list at the comment level (concise
section labels, test arithmetic, accurate summaries, invariant comments, TODO-needs-
external-intent, longer-rewrite-is-merely-nicer).

INPUT: harvested classifications.jsonl
OUTPUT: hard_negatives.jsonl -- a tagged subset, compatible with the dataset recipe.
"""
import json, collections, os, argparse

def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="classifications.jsonl")
    ap.add_argument("--out", default="hard_negatives.jsonl")
    args = ap.parse_args()

    rows = load(args.inp)

    accepted = []
    informational = []
    needs_review = []

    for r in rows:
        disp = r["disposition"]
        cls = r["classification"]
        if disp == "accepted":
            r = dict(r)
            r["hard_negative"] = {"type": "accepted", "note":
                "code pattern valid (still_valid) but developer accepted/dismissed -> do not push needs_action"}
            accepted.append(r)
        elif disp == "informational":
            r = dict(r)
            r["hard_negative"] = {"type": "informational", "note":
                "live but intentionally low-action, not requesting a fix"}
            informational.append(r)
        elif disp == "needs_review":
            r = dict(r)
            r["hard_negative"] = {"type": "needs_review", "note":
                "ambiguous/low-confidence -> defer to human, do not auto-repost"}
            needs_review.append(r)

    out_rows = accepted + informational + needs_review

    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    print("wrote", os.path.abspath(args.out))
    print("total hard negatives:", len(out_rows))
    print("  accepted      :", len(accepted))
    print("  informational :", len(informational))
    print("  needs_review  :", len(needs_review))
    print()
    rc = collections.Counter(r["metadata"]["repo"] for r in out_rows)
    print("by repo:")
    for k, v in rc.most_common():
        print("   %-28s %d" % (k, v))
    print()
    print("NOTE on WT comment-level keep-negatives:")
    print("  The umpire `accepted` cases are finding-level hard negatives. For the WT")
    print("  code-comment task, valid-as-is (`keep`) comments must ALSO be mined as")
    print("  hard negatives; those need labeling from the WT eval corpus (separate step).")

if __name__ == "__main__":
    main()
