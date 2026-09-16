#!/usr/bin/env python3
"""Decode/prefill throughput vs concurrency for an OpenAI-compatible server.

For each concurrency level C fires C concurrent /v1/completions requests with a
fixed input length (prefill) and fixed output budget (decode), and reports:
  - aggregate output (decode) tok/s across the batch
  - aggregate input (prefill) tok/s across the batch
  - mean/median per-request total latency.

Usage: python3 latency_scaling.py <url> <model> [batches]
"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8088"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "ornith"
BATCHES = int(sys.argv[3]) if len(sys.argv) > 3 else 2

INPUT_TOKENS = 256   # target prefill length
OUTPUT_TOKENS = 160  # target decode length per request
PROMPT = ("voxel pagoda garden cherry blossom lights. " * 40)  # ~40 words... adjust
# pad to roughly INPUT_TOKENS (~3.8 tok/word for this tokenizer): use repeated sentence
SENT = "the quick brown voxel fox renders a colorful pagoda in a spring garden. "
PROMPT = SENT * (INPUT_TOKENS // 15 + 1)

def oneshot(i):
    body = {"model": MODEL, "prompt": PROMPT, "max_tokens": OUTPUT_TOKENS,
            "temperature": 0.6, "top_p": 0.95}
    req = urllib.request.Request(URL + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    dt = time.time() - t0
    u = d["usage"]
    return dt, u["prompt_tokens"], u["completion_tokens"]

def run_level(C):
    # pre-warm at this concurrency once (JIT/template warm)
    with ThreadPoolExecutor(max_workers=C) as ex:
        list(ex.map(oneshot, range(C)))
    lat = []; pin = pout = 0; wall_sum = 0.0
    for _ in range(BATCHES):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=C) as ex:
            res = list(ex.map(oneshot, range(C)))
        wall = time.time() - t0
        wall_sum += wall
        for (dt, pi, co) in res:
            lat.append(dt); pin += pi; pout += co
    return len(res) * BATCHES, pin, pout, lat, wall_sum

print(f"url={URL} model={MODEL} input~{INPUT_TOKENS} out~{OUTPUT_TOKENS} batches={BATCHES}")
print(f"{'conc':>5} {'prefill tok/s':>13} {'decode tok/s':>12} {'aggr tok/s':>11} {'p50 lat':>8} {'p95 lat':>8}")
for C in (1, 2, 4, 6, 8, 12, 16):
    n, pin, pout, lat, wall_sum = run_level(C)
    prefill_tps = pin / wall_sum
    decode_tps = pout / wall_sum
    aggr_tps = (pin + pout) / wall_sum
    lat_s = sorted(lat)
    p50 = lat_s[len(lat_s)//2]; p95 = lat_s[int(len(lat_s)*0.95)-1]
    print(f"{C:5d} {prefill_tps:13.1f} {decode_tps:12.1f} {aggr_tps:11.1f} {p50*1000:7.0f}ms {p95*1000:7.0f}ms", flush=True)
