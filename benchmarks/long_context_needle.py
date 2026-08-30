#!/usr/bin/env python3
"""Needle retrieval at controlled approximate prompt-token lengths."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def filler_line(index: int) -> str:
    digest = hashlib.sha256(f"rocky-context-row-{index}".encode()).hexdigest()[:24]
    return (
        f"Archive record {index:06d}: checksum {digest}. "
        "This ordinary filler row contains no launch authorization code.\n"
    )


def make_prompt(lines: list[str], position: float, code: str) -> str:
    split = round(len(lines) * position)
    needle = (
        f"AUTHORITATIVE NEEDLE: Rocky's launch authorization code is {code}. "
        "Retain this exact code for the final question.\n"
    )
    return (
        "Read the following synthetic archive. Ignore incidental checksums.\n\n"
        + "".join(lines[:split])
        + needle
        + "".join(lines[split:])
        + "\nWhat is Rocky's launch authorization code? Reply with only the code."
    )


def token_count(args: argparse.Namespace, prompt: str) -> int:
    response = post_json(
        f"{args.url.removesuffix('/v1').rstrip('/')}/tokenize",
        {"model": args.model, "prompt": prompt},
        args.timeout,
    )
    return int(response["count"])


def build_lines(args: argparse.Namespace) -> list[str]:
    # Tokenize a small sample to estimate rows, then refine against the endpoint.
    sample = [filler_line(index) for index in range(200)]
    sample_prompt = make_prompt(sample, 0.5, "ROCKY-NEEDLE-SAMPLE")
    sample_tokens = token_count(args, sample_prompt)
    line_count = max(1, round(200 * args.target_tokens / sample_tokens))
    lines = [filler_line(index) for index in range(line_count)]

    for _ in range(4):
        count = token_count(args, make_prompt(lines, 0.5, "ROCKY-NEEDLE-SAMPLE"))
        error = args.target_tokens - count
        if abs(error) <= args.tolerance:
            break
        revised_count = max(1, round(len(lines) * args.target_tokens / count))
        if revised_count > len(lines):
            lines.extend(filler_line(index) for index in range(len(lines), revised_count))
        else:
            lines = lines[:revised_count]
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8083/v1")
    parser.add_argument("--model", default="qwen-moe")
    parser.add_argument("--target-tokens", type=int, default=60000)
    parser.add_argument("--tolerance", type=int, default=256)
    parser.add_argument("--positions", default="0.1,0.5,0.9")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run_position(
    args: argparse.Namespace, lines: list[str], ordinal: int, position: float
) -> dict:
    code = f"ROCKY-{args.target_tokens}-{ordinal}-A7F3C9"
    prompt = make_prompt(lines, position, code)
    raw_prompt_tokens = token_count(args, prompt)
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    response = post_json(
        f"{args.url.rstrip('/')}/chat/completions", payload, args.timeout
    )
    elapsed = time.perf_counter() - started
    choice = response["choices"][0]
    message = choice["message"]
    output = message.get("content") or ""
    return {
        "position": position,
        "raw_prompt_tokens": raw_prompt_tokens,
        "api_prompt_tokens": response.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": response.get("usage", {}).get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "reasoning_chars": len(message.get("reasoning") or ""),
        "expected": code,
        "output": output,
        "pass": output.strip() == code,
        "elapsed_s": round(elapsed, 3),
    }


def main() -> None:
    args = parse_args()
    lines = build_lines(args)
    jobs = list(enumerate(float(x) for x in args.positions.split(",")))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(run_position, args, lines, ordinal, position)
            for ordinal, position in jobs
        ]
        results = []
        for future in futures:
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)

    report = {
        "endpoint": args.url,
        "model": args.model,
        "target_tokens": args.target_tokens,
        "line_count": len(lines),
        "concurrency": args.concurrency,
        "passed": sum(result["pass"] for result in results),
        "total": len(results),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
