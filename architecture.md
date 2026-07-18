# Architecture

## Layers

```
domain/         pure entities, no framework imports
use_cases/      MAF ChatAgents — depend on ChatModelClient Protocol
infrastructure/ outer drivers (DB, index, scraper, chat clients, tool cache)
main.py         composition root
bench/          synthetic agent-loop benchmark
scripts/        vLLM launcher + droplet firewall lockdown
Dockerfile      ROCm 7.14.0 fallback image
```

Rules:
- `domain/` has no `infrastructure` or `agent_framework` imports
- `use_cases/` depends only on `ChatModelClient` (Protocol), never a concrete SDK
- `infrastructure/` owns all third-party libraries

## LLM client layer

`ChatModelClient` is a `@runtime_checkable` Protocol in `infrastructure/chat_clients.py`.
Any class with `model: str` and `complete(turn: ChatTurn) -> str` implements it.

Two concrete providers:

| Provider | When | Endpoint |
|---|---|---|
| `VLLMClient` | AMD Dev Cloud MI300X (1-Click image) | `http://localhost:8000/v1` |
| `DeepSeekClient` | Local dev (no GPU) | `https://api.deepseek.com/v1` |

Both speak the OpenAI Chat Completions wire protocol — agent code is identical regardless of backend.

## Agents

| Agent | Role | Tools |
|---|---|---|
| `DiscoveryAgent` | Extract structured JSON brief from free-form user input | none |
| `SynthesisAgent` | Rank candidates against the brief | search_catalog, fetch_product_page, fetch_html |

## Tool layer

`build_tools()` in `infrastructure/agent_tools.py` returns three MAF-registered tools:
- `search_catalog(query, limit)` — BM25 exact-keyword over local SQLite
- `fetch_product_page(url, selectors)` — Playwright parse to ProductPayload
- `fetch_html(url, wait_selector)` — raw HTML with optional wait

All tool results are cached in-process via `cachetools.TTLCache(maxsize=512, ttl=300)`.

## Storage

- **SQLite** (`./retail_catalog.db`) — products table with `structured_data` JSON column, queried via `json_extract` (SQLite JSON1)
- **BM25Okapi corpus** — rebuilt on startup from the SQLite catalog, kept in-memory
- **Playwright profile** — persistent chromium context at `/home/rajasingh/.mozilla/edge_profile`

All state lives on the AMD GPU droplet's local NVMe. No external object store.

## Data flow

```
user message
   │
   ▼
DiscoveryAgent  ──► JSON brief (intent, brands, budget_max, must_have, …)
   │
   ▼
SynthesisAgent  ──► tool calls (search_catalog → BM25 → SQLite)
                              (fetch_product_page → Playwright → live sites)
   │
   ▼
ranked recommendations with rationale
```

## Bench evidence

`bench/run_agent_bench.py` runs 5 Discovery→Synthesis turns and writes a JSON report to `bench/results/agent_bench_<timestamp>.json`. Captures:

- per-turn latency (mean / median / best / worst)
- tool-result cache effectiveness (size after last turn)
- `rocm-smi` / `amd-smi` snapshots before & after
- vLLM prefix-cache hit rate before & after (the headline AMD rubric number)