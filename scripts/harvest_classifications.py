#!/usr/bin/env python3
"""
Harvest umpire-bot rereview classification data into a training-ready JSONL dataset.

Sources (all local, no network):
  1. /var/lib/pr-reviewer/classifications/**/*.json  -> adjudicated classification per comment_id
  2. state.db  -> pr_inline_comments (finding body/path/line/repo/pr) and finding_feedback
                  (reply_sentiment + reaction signals for discussion evidence)

Output: one JSON object per line, shape compatible with the post-training spike plan
(messages[] + metadata). The assistant message carries the strict classification JSON.

Note: current source code / diff excerpt are NOT stored in the DB. This harvest emits
finding-level features (finding body, path, line) and the adjudicated verdict. Enrichment
with real source bundles from git worktrees is a phase-2 step (see plan).
"""
import json, glob, os, sqlite3, collections, sys, argparse

DEFAULT_DB = "/var/lib/pr-reviewer/state.db"
DEFAULT_CLS = "/var/lib/pr-reviewer/classifications"


def load_comment_index(db_path):
    """comment_github_id -> inline comment row"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    idx = {}
    rows = conn.execute("SELECT * FROM pr_inline_comments").fetchall()
    for r in rows:
        idx[r["comment_github_id"]] = dict(r)
    conn.close()
    return idx


def load_feedback(db_path):
    """comment_github_id -> list of feedback signals"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    by_comment = collections.defaultdict(list)
    rows = conn.execute(
        "SELECT * FROM finding_feedback WHERE feedback_type IN ('reply_sentiment','reaction')"
    ).fetchall()
    for r in rows:
        by_comment[r["comment_github_id"]].append(dict(r))
    conn.close()
    return by_comment


def build_discussion(feedback_rows):
    """Reconstruct a discussion_replies-style block from stored signals."""
    if not feedback_rows:
        return None
    disc = []
    for f in feedback_rows:
        if f["feedback_type"] == "reply_sentiment":
            disc.append({"kind": "reply_sentiment", "signal": f["feedback_signal"],
                         "confidence": f["feedback_strength"], "detail": f.get("feedback_detail")})
        elif f["feedback_type"] == "reaction":
            disc.append({"kind": "reaction", "signal": f["feedback_signal"],
                         "count": f["feedback_strength"]})
    return disc if disc else None


def classification_entries(cls_root):
    for fp in sorted(glob.glob(os.path.join(cls_root, "**", "*.json"), recursive=True)):
        rel = os.path.relpath(fp, cls_root)
        parts = rel.split(os.sep)
        owner_repo = parts[0] + "/" + parts[1]
        try:
            data = json.load(open(fp))
        except Exception as e:
            print("WARN parse fail", fp, e, file=sys.stderr)
            continue
        for e in data.get("entries", []):
            yield owner_repo, fp, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--cls", default=DEFAULT_CLS)
    ap.add_argument("--out", default="umpire-classifications.jsonl")
    ap.add_argument("--no-orphans", action="store_true",
                    help="skip entries whose finding body is not in the DB")
    args = ap.parse_args()

    comments = load_comment_index(args.db)
    feedback = load_feedback(args.db)

    stats = collections.Counter()
    out_rows = []
    with open(args.out, "w") as f:
        for owner_repo, fp, e in classification_entries(args.cls):
            cid = e.get("comment_id")
            cl = e.get("classification") or {}
            v = cl.get("validation") or {}
            if not v.get("valid"):
                stats["skipped_invalid"] += 1
                continue
            cmt = comments.get(cid)
            if not cmt:
                if args.no_orphans:
                    stats["skipped_orphan"] += 1
                    continue
            stats["valid"] += 1
            disc = build_discussion(feedback.get(cid))
            meta = {
                "repo": cmt["repo"] if cmt else owner_repo,
                "pr_number": cmt["pr_number"] if cmt else None,
                "review_github_id": cmt["review_github_id"] if cmt else None,
                "comment_id": cid,
                "path": (cmt["path"] if cmt else None) or None,
                "line": cmt["line"] if cmt else None,
                "head_sha": cmt["head_sha"] if cmt else None,
                "classification_source": fp,
                "prior_finding_id": cl.get("prior_finding_id"),
                "confidence": cl.get("confidence"),
                "validation": v,
            }
            row = {
                "classification": cl.get("classification"),
                "disposition": cl.get("disposition"),
                "evidence": cl.get("evidence"),
                "disposition_evidence": cl.get("disposition_evidence"),
                "current_locations": cl.get("current_locations"),
                "finding_body": cmt["body"] if cmt else None,
                "discussion": disc,
                "metadata": meta,
            }
            out_rows.append(row)
            f.write(json.dumps(row) + "\n")

    print("Wrote", os.path.abspath(args.out))
    print("total valid rows:", stats["valid"])
    print("skipped invalid (validation.valid=false):", stats["skipped_invalid"])
    print("skipped orphan (no body in DB):", stats.get("skipped_orphan", 0))
    print("\nclass/disposition breakdown:")
    cc = collections.Counter(r["classification"] for r in out_rows)
    dd = collections.Counter(r["disposition"] for r in out_rows)
    for k, v in cc.most_common():
        print("  class %-12s %d" % (k, v))
    for k, v in dd.most_common():
        print("  disp  %-14s %d" % (k, v))
    print("\nrows with finding_body:", sum(1 for r in out_rows if r["finding_body"]))
    print("rows with discussion evidence:", sum(1 for r in out_rows if r["discussion"]))


if __name__ == "__main__":
    main()
