#!/usr/bin/env python3
"""Quick batch-concurrency benchmark against the Radeon vLLM endpoint.

Measures decode tokens/sec at concurrency=1 (single) and concurrency=4 (batched)
to calculate the throughput increase from vLLM's continuous batching.
"""
import asyncio, json, os, time, statistics
import httpx

BASE_URL = os.getenv("RETAIL_BASE_URL", "http://129.212.178.184:8000/v1")
MODEL = os.getenv("RETAIL_MODEL", "google/gemma-4-31B-it")
PROMPT = "Write a short product description for an ergonomic office chair. Include benefits like lumbar support, adjustable height, and breathable mesh material."
GEN_TOKENS = 256

async def one_request(client, idx):
    """Send one chat request. Returns timing info."""
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    async with client.stream(
        "POST",
        f"{BASE_URL.rstrip('/')}/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": GEN_TOKENS,
            "temperature": 0,
            "stream": True,
        },
        timeout=300,
    ) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            if line.strip() == "data: [DONE]":
                break
            try:
                d = json.loads(line[5:])
                if d["choices"][0]["delta"].get("content"):
                    ntok += 1
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
    dur = time.perf_counter() - t0
    decode_tps = ntok / max(1e-6, dur - ttft) if dur > ttft else 0
    return {"idx": idx, "ttft": ttft, "e2e": dur, "gen_tok": ntok, "decode_tps": decode_tps}

async def bench_concurrency(concurrency, n_requests):
    """Run n_requests at a given concurrency. Returns summary."""
    print(f"\n--- Concurrency={concurrency} ({n_requests} requests) ---")
    t0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        tasks = [one_request(client, i) for i in range(n_requests)]
        sem = asyncio.Semaphore(concurrency)
        async def limited(task):
            async with sem:
                return await task
        results = await asyncio.gather(*[limited(t) for t in tasks])
    elapsed = time.perf_counter() - t0

    ttfts = [r["ttft"] for r in results if r["ttft"]]
    e2es = [r["e2e"] for r in results]
    d_tps = [r["decode_tps"] for r in results]
    gen_toks = [r["gen_tok"] for r in results]
    total_toks = sum(gen_toks)

    system_tps = total_toks / max(1e-6, max(e2es))

    print(f"  Wall time: {elapsed:.2f}s")
    print(f"  Per-request mean TTFT: {statistics.mean(ttfts):.3f}s" if ttfts else "  No TTFT data")
    print(f"  Per-req mean decode: {statistics.mean(d_tps):.1f} tok/s")
    print(f"  System throughput: {system_tps:.1f} tok/s")
    print(f"  Total tokens: {total_toks}")
    print(f"  Mean per-req e2e: {statistics.mean(e2es):.2f}s")
    print(f"  Max per-req e2e: {max(e2es):.2f}s")
    return system_tps

async def main():
    print(f"Endpoint: {BASE_URL}")
    print(f"Model: {MODEL}")
    print(f"Prompt: {PROMPT[:60]}...")
    print(f"Gen tokens: {GEN_TOKENS}")
    print()

    tps_single = await bench_concurrency(1, 3)
    tps_batch4 = await bench_concurrency(4, 8)
    tps_batch8 = await bench_concurrency(8, 16)

    print(f"\n{'='*50}")
    print(f"Single (1):      {tps_single:.1f} tok/s")
    print(f"Batched (4):     {tps_batch4:.1f} tok/s")
    print(f"Batched (8):     {tps_batch8:.1f} tok/s")
    if tps_single > 0:
        print(f"Speedup 4x:      {tps_batch4 / tps_single:.2f}x")
        print(f"Speedup 8x:      {tps_batch8 / tps_single:.2f}x")

if __name__ == "__main__":
    asyncio.run(main())
