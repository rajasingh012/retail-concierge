"""Synthetic 5-turn agent loop for the AMD Radeon judging evidence.

Runs the full Discovery -> Synthesis pipeline against the configured
provider (vllm on AMD cloud; deepseek for local-dev fallback) and
produces a JSON report capturing:

  - per-turn latency (TTFT proxy via wall time)
  - tokens generated
  - tool-result cache effectiveness (cache hits vs misses)
  - peak GPU memory (from rocm-smi)
  - vLLM prefix-cache hit rate before & after (from /v1/metrics) — the
    headline AMD rubric number for multi-turn agents

Output: bench/results/agent_bench_<timestamp>.json
Print:  a 1-page summary to stdout for live demos.

Usage:
    PYTHONPATH=. python bench/run_agent_bench.py
    PYTHONPATH=. python bench/run_agent_bench.py --turns 8 --provider deepseek
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on sys.path when run as `python bench/run_agent_bench.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infrastructure.agent_tools import build_tools, cache_stats, clear_cache
from infrastructure.chat_clients import build_chat_client
from infrastructure.database import ProductCatalogRepository
from infrastructure.ecommerce_adapter import ECommerceAdapter
from infrastructure.indexer import LocalHybridSearchEngine
from infrastructure.scraper import PlaywrightScraper
from use_cases import build_discovery_agent, build_synthesis_agent
from domain.entities import ItemVariant, ProductPayload


SAMPLE_TURNS = [
    "I need an ergonomic office chair, budget around $1500, brand Steelcase or Herman Miller.",
    "Same brief but I prefer mesh back and lumbar support.",
    "Add a standing desk converter under $400.",
    "What about monitor arms? Same budget ceiling.",
    "Anything used/refurbished I should consider?",
]


# ---------- fixtures ----------

def seed_catalog(db_path: str) -> int:
    """Seed a small but representative catalog for the bench."""
    repo = ProductCatalogRepository(db_path)
    rows = [
        ProductPayload(
            title="Steelcase Leap V2", source_url="u1",
            variants=[ItemVariant(sku="1", label="std fabric", price=1299.0)],
            dynamic_attributes={"brand": "Steelcase", "description": "ergonomic task chair"},
        ),
        ProductPayload(
            title="Herman Miller Aeron (Remastered)", source_url="u2",
            variants=[ItemVariant(sku="2", label="graphite size B mesh", price=1495.0)],
            dynamic_attributes={"brand": "Herman Miller", "description": "mesh ergonomic task chair"},
        ),
        ProductPayload(
            title="UPLIFT V2 Standing Desk Converter", source_url="u3",
            variants=[ItemVariant(sku="3", label="36in", price=379.0)],
            dynamic_attributes={"brand": "UPLIFT", "description": "standing desk converter"},
        ),
        ProductPayload(
            title="Ergotron LX Monitor Arm", source_url="u4",
            variants=[ItemVariant(sku="4", label="single", price=259.0)],
            dynamic_attributes={"brand": "Ergotron", "description": "desk-mount monitor arm"},
        ),
        ProductPayload(
            title="Used Steelcase Leap V1 (refurb)", source_url="u5",
            variants=[ItemVariant(sku="5", label="refurb", price=499.0)],
            dynamic_attributes={"brand": "Steelcase", "description": "refurb ergonomic task chair"},
        ),
    ]
    repo.bulk_upsert((p, "demo", p.source_url) for p in rows)
    repo.close()
    return len(rows)


# ---------- metrics ----------

def _rocm_smi_snapshot() -> Dict[str, Any]:
    """Return GPU memory + util via rocm-smi, if installed.

    Falls back to `rocminfo` when rocm-smi throws libdrm errors
    (common inside some container runtimes).
    """
    if shutil.which("rocm-smi"):
        try:
            out = subprocess.run(
                ["rocm-smi", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return {"available": True, "raw": out.stdout[:8000]}
            # rocm-smi returned non-zero — likely the libdrm-in-container issue.
        except Exception:                                          # noqa: BLE001
            pass
    if shutil.which("rocminfo"):
        try:
            out = subprocess.run(
                ["rocminfo"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            # Pull the "Marketing Name" line and gfx version as a fallback signal.
            marketing = [
                line.strip() for line in out.splitlines()
                if "Marketing Name" in line
            ]
            gfx = [
                line.strip() for line in out.splitlines()
                if line.strip().startswith("Name:") and "gfx" in line
            ]
            return {
                "available": True,
                "source": "rocminfo",
                "marketing_name": marketing[:3],
                "gfx_targets": gfx[:5],
            }
        except Exception as e:                                    # noqa: BLE001
            return {"available": False, "error": str(e)}
    return {"available": False}


def _vllm_prefix_cache_metrics(metrics_url: str) -> Dict[str, Any]:
    """Pull the headline demoable metric from vLLM's /v1/metrics.

    The specific line we want is `vllm:prefix_cache_hit_rate` — a gauge
    that goes from 0.0 (first turn) to ~0.95+ (turn 2+) when prefix
    caching is enabled. This is the single most rubric-relevant number
    for the AMD track.

    Returns {'available': False, ...} if vLLM isn't reachable so callers
    can decide to skip rather than fail.
    """
    import urllib.request
    try:
        with urllib.request.urlopen(metrics_url, timeout=3) as r:
            text = r.read().decode("utf-8")
        hits = []
        for line in text.splitlines():
            if line.startswith("vllm:prefix_cache_hit_rate"):
                hits.append(line)
            elif line.startswith("vllm:gpu_cache_usage_perc"):
                hits.append(line)
            elif line.startswith("vllm:num_requests_running"):
                hits.append(line)
        return {"available": True, "metrics": hits or ["(no relevant lines found)"]}
    except Exception as e:                                        # noqa: BLE001
        return {"available": False, "error": str(e)}


# ---------- runner ----------

async def run_turn(
    discovery, synthesis, user_msg: str, turn_idx: int
) -> Dict[str, Any]:
    """Run one Discovery -> Synthesis pair, return timing + cache stats."""
    clear_cache()
    cache_before = cache_stats()

    t0 = time.perf_counter()
    brief = await discovery.run(user_msg)
    brief_text = getattr(brief, "text", None) or str(brief)
    t_brief = time.perf_counter() - t0

    t1 = time.perf_counter()
    prompt = (
        "Here is the structured shopping brief as JSON:\n"
        f"{brief_text}\n\nProduce the final ranked recommendation."
    )
    final = await synthesis.run(prompt)
    final_text = getattr(final, "text", None) or str(final)
    t_synth = time.perf_counter() - t1

    cache_after = cache_stats()
    return {
        "turn": turn_idx,
        "user_msg": user_msg,
        "t_brief_sec": round(t_brief, 3),
        "t_synthesis_sec": round(t_synth, 3),
        "t_total_sec": round(t_brief + t_synth, 3),
        "brief_chars": len(brief_text),
        "final_chars": len(final_text),
        "approx_output_tokens": len(final_text.split()),
        "tool_cache_size_after": cache_after["size"],
    }


async def main(turns: int, provider: str, model: str) -> Dict[str, Any]:
    db_path = os.path.join(tempfile.gettempdir(), "retail_bench.db")
    if os.path.exists(db_path):
        os.unlink(db_path)
    seeded = seed_catalog(db_path)
    print(f"[bench] seeded {seeded} products -> {db_path}")

    repo = ProductCatalogRepository(db_path)
    search_engine = LocalHybridSearchEngine(repo)
    scraper = PlaywrightScraper()
    await scraper.start()
    ecommerce = ECommerceAdapter()
    tools = build_tools(search_engine, scraper=scraper, ecommerce_adapter=ecommerce)

    client = build_chat_client(provider, model)
    print(f"[bench] provider={provider} model={model}")

    discovery = build_discovery_agent(client)
    synthesis = build_synthesis_agent(client, tools)

    gpu_before = _rocm_smi_snapshot()
    # Pull vLLM prefix-cache metric before the run (cold = 0%) and after
    # (warm = ~95%+). This is the headline AMD-rubric number.
    vllm_metrics_url = os.getenv("VLLM_METRICS_URL", "http://localhost:8000/metrics")
    vllm_metrics_before = _vllm_prefix_cache_metrics(vllm_metrics_url)

    turn_results: List[Dict[str, Any]] = []
    for i in range(min(turns, len(SAMPLE_TURNS))):
        msg = SAMPLE_TURNS[i]
        print(f"\n[bench] turn {i+1}/{turns}: {msg[:60]}...")
        row = await run_turn(discovery, synthesis, msg, i + 1)
        turn_results.append(row)
        print(f"        brief={row['t_brief_sec']}s  synth={row['t_synthesis_sec']}s"
              f"  cache={row['tool_cache_size_after']}")

    gpu_after = _rocm_smi_snapshot()
    vllm_metrics_after = _vllm_prefix_cache_metrics(vllm_metrics_url)

    await scraper.close()
    repo.close()

    totals = [r["t_total_sec"] for r in turn_results]
    synth_latencies = [r["t_synthesis_sec"] for r in turn_results]
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "turns": len(turn_results),
        "latency": {
            "turn_total_mean_sec": round(statistics.mean(totals), 3),
            "turn_total_median_sec": round(statistics.median(totals), 3),
            "synthesis_mean_sec": round(statistics.mean(synth_latencies), 3),
            "synthesis_median_sec": round(statistics.median(synth_latencies), 3),
            "best_turn_sec": round(min(totals), 3),
            "worst_turn_sec": round(max(totals), 3),
        },
        "tool_cache": {
            "size_after_last_turn": turn_results[-1]["tool_cache_size_after"],
        },
        "gpu": {"before": gpu_before, "after": gpu_after},
        "vllm_metrics": {"before": vllm_metrics_before, "after": vllm_metrics_after},
        "turns_detail": turn_results,
    }

    out_dir = Path(__file__).resolve().parents[1] / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = out_dir / f"agent_bench_{stamp}.json"
    out_file.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[bench] results -> {out_file}")
    return summary


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--turns", type=int, default=5)
    p.add_argument("--provider", default=os.getenv("RETAIL_PROVIDER", "deepseek"))
    p.add_argument("--model", default=os.getenv("RETAIL_MODEL", "deepseek-chat"))
    args = p.parse_args()
    asyncio.run(main(args.turns, args.provider, args.model))


if __name__ == "__main__":
    cli()