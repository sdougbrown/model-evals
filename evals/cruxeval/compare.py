#!/usr/bin/env python3
"""
Compare model output to expected CRUXEval value using ast.literal_eval.
Reads JSON from stdin: {"got": "...", "exp": "..."}
Prints "pass" or "fail".
"""
import ast
import json
import re
import sys


def extract_candidate(raw: str) -> str:
    # Strip triple-backtick fences
    raw = re.sub(r'^```\w*\n?', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\n?```$', '', raw, flags=re.MULTILINE)
    raw = raw.strip()
    # Strip inline backtick wrapping: `value` → value
    raw = re.sub(r'^`([^`]+)`$', r'\1', raw).strip()
    # If multi-line, take the last non-empty line (model may add preamble)
    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    return lines[-1] if lines else raw


def main():
    data = json.load(sys.stdin)
    got_raw = data['got']
    exp_raw = data['exp']

    candidate = extract_candidate(got_raw)
    expected_str = exp_raw.strip()

    try:
        got_val = ast.literal_eval(candidate)
    except Exception:
        print('fail')
        return

    try:
        exp_val = ast.literal_eval(expected_str)
    except Exception:
        # Expected is malformed — fall back to string equality
        print('pass' if candidate == expected_str else 'fail')
        return

    print('pass' if got_val == exp_val else 'fail')


if __name__ == '__main__':
    main()
