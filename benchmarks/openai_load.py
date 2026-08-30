#!/usr/bin/env python3
"""Small dependency-free load benchmark for OpenAI-compatible chat endpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RequestResult:
    latency_s: float
    ttft_s: float
    completion_tokens: int
    prompt_tokens: int

    @property
    def decode_tokens_per_s(self) -> float:
        decode_s = max(self.latency_s - self.ttft_s, 1e-9)
        # The first token arrives at TTFT, so only later tokens occupy decode_s.
        return max(self.completion_tokens - 1, 0) / decode_s


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(math.ceil(fraction * len(ordered)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def run_request(args: argparse.Namespace) -> RequestResult:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "reasoning_effort": "medium" if args.thinking else "none",
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {
            "enable_thinking": args.thinking,
            "reasoning": args.thinking,
        },
    }
    if args.ignore_eos:
        payload["ignore_eos"] = True

    request = urllib.request.Request(
        f"{args.url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
    )
    started = time.perf_counter()
    first_token_at = None
    usage = {}

    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if first_token_at is None and (
                    delta.get("content") or delta.get("reasoning")
                ):
                    first_token_at = time.perf_counter()

    finished = time.perf_counter()
    if first_token_at is None:
        first_token_at = finished
    return RequestResult(
        latency_s=finished - started,
        ttft_s=first_token_at - started,
        completion_tokens=int(usage.get("completion_tokens", 0)),
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
    )


def benchmark_level(args: argparse.Namespace, concurrency: int) -> dict:
    requests = args.requests or max(4, concurrency * 2)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_request, args) for _ in range(requests)]
        results = [future.result() for future in futures]
    wall_s = time.perf_counter() - started

    latencies = [item.latency_s for item in results]
    ttfts = [item.ttft_s for item in results]
    decode_rates = [item.decode_tokens_per_s for item in results]
    completion_tokens = sum(item.completion_tokens for item in results)
    return {
        "concurrency": concurrency,
        "requests": requests,
        "wall_s": round(wall_s, 3),
        "prompt_tokens_per_request": results[0].prompt_tokens,
        "completion_tokens": completion_tokens,
        "aggregate_output_tokens_per_s": round(completion_tokens / wall_s, 2),
        "request_latency_s_p50": round(statistics.median(latencies), 3),
        "request_latency_s_p95": round(percentile(latencies, 0.95), 3),
        "ttft_s_p50": round(statistics.median(ttfts), 3),
        "ttft_s_p95": round(percentile(ttfts, 0.95), 3),
        "per_request_decode_tokens_per_s_p50": round(
            statistics.median(decode_rates), 2
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8083/v1")
    parser.add_argument("--model", default="qwen-moe")
    parser.add_argument("--concurrency", default="1,2,4,6")
    parser.add_argument("--requests", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--prompt",
        default=(
            "Explain how a work-stealing task scheduler operates, including queue "
            "ownership, stealing, synchronization, shutdown, and failure handling."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels = [int(value) for value in args.concurrency.split(",")]
    if not args.no_warmup:
        warmup_args = argparse.Namespace(**vars(args))
        warmup_args.max_tokens = min(args.max_tokens, 64)
        run_request(warmup_args)

    report = {
        "endpoint": args.url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "ignore_eos": args.ignore_eos,
        "levels": [],
    }
    for level in levels:
        result = benchmark_level(args, level)
        report["levels"].append(result)
        print(json.dumps(result), flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
