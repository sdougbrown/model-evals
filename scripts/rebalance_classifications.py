#!/usr/bin/env python3
"""
Rebalance plan + repo-split tool for the harvested umpire classification corpus.

IMPORTANT HONEST NOTE: the raw harvested set (348 rows) is NOT big enough to produce a
balanced 1,800-example first-adapter training set by resampling alone -- there is no
synthetic generator here, and dropping the dominant `fixed` majority would hurt precision
(we DO want `fixed`/no_action examples; we just do not want them to swamp live classes).

So this tool does two things that ARE useful with the data we have:
  1. Repo-hygienic split (whole repos -> train / holdout) -- the manifest.
  2. A GAP REPORT: given a target first-adapter composition (~1,800) and the class mix
     each source should contribute, state exactly how many rows of each class are still
     needed and from which bucket (mined hard negatives, more annotated inlines, prose,
     etc). This is the draft rebalancing plan.

It does NOT fabricate rows. When the additional sources (hard negatives, annotated
inlines, prose) exist, re-run to slice them into the same balanced slots.
"""
import json, collections, os, argparse

TRAIN_REPOS = {"sdougbrown/avenor", "sdougbrown/umpire-bot"}
HOLDOUT_REPOS = {"sdougbrown/writetighter", "sdougbrown/daywatch-cal",
                 "umpire-tools/umpire-go-gen"}

# Target composition for a first useful adapter (from the plan, section 3)
TARGET = [
    ("rebalanced_harvest",  500),
    ("hard_negatives",      400),
    ("annotated_inlines",   350),
    ("prose_revision",      400),
    ("needs_context",       150),
]
TARGET_TOTAL = sum(n for _, n in TARGET)

def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="classifications.jsonl")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--target", type=int, default=1800)
    args = ap.parse_args()

    rows = [r for r in load(args.inp) if r["classification"] != "uncertain"]

    # Repo split
    def split_of(r):
        repo = r["metadata"]["repo"]
        return "holdout" if repo in HOLDOUT_REPOS else "train"
    for r in rows:
        r["_split"] = split_of(r)

    by_split = collections.defaultdict(list)
    for r in rows:
        by_split[r["_split"]].append(r)

    by_split_class = {
        s: collections.Counter(r["classification"] for r in by_split[s])
        for s in by_split
    }

    # Use the actual available rows first (do not fabricate)
    train_rows = by_split["train"]
    print("=" * 62)
    print("HONEST STATE")
    print("=" * 62)
    print("harvested classified rows (uncertain excluded):", len(rows))
    print("  train  :", len(by_split["train"]))
    print("  holdout:", len(by_split["holdout"]))
    print()
    print("train class mix:")
    for k, v in by_split_class["train"].most_common():
        print("   %-12s %d" % (k, v))
    print()
    print("=" * 62)
    print("GAP REPORT vs first-adapter target")
    print("=" * 62)
    print("target total: %d  (available harvested: %d)" % (TARGET_TOTAL, len(rows)))
    print()
    # The 'rebalanced_harvest' bucket is the only one we can partly fill today.
    avail = len(rows)
    need_from_harvest = dict(TARGET)["rebalanced_harvest"]
    harvest_gap = max(0, need_from_harvest - avail)
    print("rebalanced_harvest bucket target: %d  -> have %d  -> gap %d" % (
        need_from_harvest, avail, harvest_gap))
    print("  (cap the fixed share here; oversample still_valid/obsolete in the slice")
    print("   you select — see slice_hints below)")
    print()
    print("other buckets are entirely to be collected/generated:")
    for name, n in TARGET:
        if name == "rebalanced_harvest":
            continue
        print("   %-22s target %4d  have 0 (add from: %s)" % (name, n, SRC.get(name, "")))
    print()
    print("=" * 62)
    print("SLICE HINTS for the rebalanced_harvest bucket")
    print("=" * 62)
    print("To hit ~%d balanced classification examples WITHOUT synthetic data:" % need_from_harvest)
    print("  - fixed       : take ~%d  (keeps precision/no_action coverage)" % min(need_from_harvest//2, by_split_class["train"]["fixed"]))
    print("  - still_valid : take ALL %d (the live-materiality calibration signal)" % by_split_class["train"]["still_valid"])
    print("  - obsolete    : take ALL %d" % by_split_class["train"]["obsolete"])
    print("  - remaining slack (%d) -> must come from newly collected/annotated data" % max(0, need_from_harvest - (by_split_class["train"]["fixed"] + by_split_class["train"]["still_valid"] + by_split_class["train"]["obsolete"])))
    print()
    # Write manifest
    split_manifest = {
        "total_harvested": len(rows),
        "target_first_adapter": TARGET_TOTAL,
        "split": {k: len(v) for k, v in by_split.items()},
        "train_class_mix": dict(by_split_class["train"]),
        "holdout_class_mix": dict(by_split_class["holdout"]),
        "buckets_target": dict(TARGET),
        "rebalanced_harvest_gap": harvest_gap,
    }
    out = os.path.join(args.out_dir, "rebalance_manifest.json")
    with open(out, "w") as f:
        json.dump(split_manifest, f, indent=2)
    # write train/holdout jsonl too
    for s, lst in by_split.items():
        p = os.path.join(args.out_dir, "split_%s.jsonl" % s)
        with open(p, "w") as f:
            for r in lst:
                # strip internal _split
                r = {kk: vv for kk, vv in r.items() if kk != "_split"}
                f.write(json.dumps(r) + "\n")
    print()
    print("wrote manifest:", out)
    print("wrote split_train.jsonl / split_holdout.jsonl")

SRC = {
    "hard_negatives": "mine from existing still_valid/accepted + WT expected-findings.json",
    "annotated_inlines": "annotate pr_inline_comments (reuse finding_feedback signals for triage)",
    "prose_revision": "WT revise-domain prose examples (Adapter B)",
    "needs_context": "findings needing non-local context",
}

if __name__ == "__main__":
    main()
