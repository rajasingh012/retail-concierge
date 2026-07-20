"""Benchmark the Discovery → Research → Critic agent collaboration."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infrastructure.agent_tools import build_tools, cache_stats, clear_cache
from infrastructure.chat_clients import build_chat_client
from infrastructure.database import ProductCatalogRepository
from use_cases import build_critic_agent, build_discovery_agent, build_research_agent
from use_cases.collaboration import run_collaboration

SAMPLE_SCENARIOS = [
    "I need a lightweight carry-on spinner luggage under $180 with at least 4 stars.",
    "Find noise cancelling over-ear headphones under $250 with strong ratings.",
    "I need a 27-inch computer monitor under $300 for office work.",
    "Recommend a laptop backpack under $80 for daily commuting.",
    "Find a highly rated mechanical gaming keyboard under $120.",
]


async def _bench_clarification_answer(_: str) -> str:
    return "Use reasonable defaults and continue with the available constraints."


def _gpu_snapshot() -> dict[str, Any]:
    """Capture a compact AMD GPU snapshot when host tooling is available."""
    for command in (["amd-smi", "metric", "--json"], ["rocm-smi", "--json"]):
        if not shutil.which(command[0]):
            continue
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=8)
            if result.returncode == 0:
                return {"available": True, "source": command[0], "raw": result.stdout[:8000]}
        except Exception as exc:
            return {"available": False, "error": str(exc)}
    return {"available": False}


def _vllm_metrics(url: str) -> dict[str, Any]:
    """Capture relevant vLLM cache and request metrics without failing the bench."""
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            lines = response.read().decode("utf-8").splitlines()
        prefixes = (
            "vllm:prefix_cache", "vllm:gpu_cache_usage",
            "vllm:num_requests_running", "vllm:time_to_first_token",
        )
        selected = [line for line in lines if line.startswith(prefixes)]
        return {"available": True, "metrics": selected}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    repository = ProductCatalogRepository(args.database)
    stats = repository.stats()
    client = build_chat_client(args.provider, args.model)
    tools = build_tools(repository)
    discovery = build_discovery_agent(client)
    research = build_research_agent(client, tools)
    critic = build_critic_agent(client)
    scenarios = SAMPLE_SCENARIOS[: max(1, min(args.turns, len(SAMPLE_SCENARIOS)))]
    metrics_url = os.getenv("VLLM_METRICS_URL", "http://localhost:8000/metrics")

    clear_cache()
    gpu_before = _gpu_snapshot()
    vllm_before = _vllm_metrics(metrics_url)
    rows: list[dict[str, Any]] = []
    try:
        for index, scenario in enumerate(scenarios, start=1):
            before = cache_stats()
            started = time.perf_counter()
            result = await run_collaboration(
                discovery, research, critic, scenario, _bench_clarification_answer
            )
            elapsed = time.perf_counter() - started
            after = cache_stats()
            row = {
                "scenario": index,
                "request": scenario,
                "latency_sec": round(elapsed, 3),
                "clarifications_requested": result.clarifications_requested,
                "candidates_researched": len(result.research.get("candidates", [])),
                "recommendations": len(result.recommendation.get("ranked", [])),
                "cache_hits_delta": after["hits"] - before["hits"],
                "cache_misses_delta": after["misses"] - before["misses"],
            }
            rows.append(row)
            print(
                f"[bench] {index}/{len(scenarios)} {row['latency_sec']}s; "
                f"evidence={row['candidates_researched']} ranked={row['recommendations']}"
            )
    finally:
        repository.close()

    gpu_after = _gpu_snapshot()
    vllm_after = _vllm_metrics(metrics_url)
    latencies = [row["latency_sec"] for row in rows]
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "model": args.model,
        "catalog": stats,
        "scenarios": len(rows),
        "latency": {
            "mean_sec": round(statistics.mean(latencies), 3),
            "median_sec": round(statistics.median(latencies), 3),
            "best_sec": min(latencies),
            "worst_sec": max(latencies),
        },
        "tool_cache": cache_stats(),
        "gpu": {"before": gpu_before, "after": gpu_after},
        "vllm_metrics": {"before": vllm_before, "after": vllm_after},
        "details": rows,
    }
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = output_dir / f"agent_bench_{stamp}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[bench] report: {output}")
    return report


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--database", type=Path, default=Path("retail_catalog.db"))
    parser.add_argument("--provider", default=os.getenv("RETAIL_PROVIDER", "vllm"))
    parser.add_argument(
        "--model", default=os.getenv("RETAIL_MODEL", "google/gemma-3-27b-it")
    )
    asyncio.run(run_benchmark(parser.parse_args()))


if __name__ == "__main__":
    cli()
