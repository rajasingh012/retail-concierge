# Architecture

## Layers

```
domain/         pure entities, no framework imports
use_cases/      MAF Agents — depend on OpenAIChatClient
infrastructure/ outer drivers (DB, index, chat clients, tool cache,
                ecommerce adapter)
vendor/         git submodule — E-Commerces-WebScraper (Amazon + other sites;
                we use Amazon only for this hackathon)
main.py         composition root
bench/          synthetic agent-loop benchmark
scripts/        vLLM launcher + droplet firewall lockdown
Dockerfile      ROCm 7.14.0 fallback image
```

Rules:
- `domain/` has no `infrastructure` or `agent_framework` imports
- `use_cases/` depends only on `OpenAIChatClient`, never a concrete SDK
- `infrastructure/` owns all third-party libraries

## LLM client layer

The chat client is `agent_framework.openai.OpenAIChatClient` (Microsoft Agent
Framework's native OpenAI-protocol client). Both backends speak the OpenAI
Chat Completions wire protocol, so swapping them is just a `base_url` change
in the factory — no code change in agents or tools.

| Provider   | When                                  | Endpoint                          |
|------------|---------------------------------------|-----------------------------------|
| `vllm`     | AMD Dev Cloud MI300X (1-Click image)  | `http://localhost:8000/v1`        |
| `deepseek` | Local dev (no GPU)                    | `https://api.deepseek.com/v1`     |

Defined in `infrastructure/chat_clients.py` (single `build_chat_client(provider, model, **kw)`
factory backed by a `PROVIDERS` registry).

## Agents

| Agent | Role | Tools |
|---|---|---|
| `DiscoveryAgent` | Extract structured JSON brief from free-form user input | none |
| `SynthesisAgent` | Rank candidates against the brief | search_catalog, fetch_product_from_site |

## Tool layer

`build_tools()` in `infrastructure/agent_tools.py` registers these tools with MAF:

- `search_catalog(query, limit)` — BM25 exact-keyword over local SQLite
- `fetch_product_from_site(url, platform="amazon")` — Live Amazon lookup via
  the `vendor/ecommerce-scraper` submodule. The submodule also includes
  AliExpress / Shein / Shopee / Mercado Livre scrapers; this hackathon
  uses Amazon only. Pass a different `platform=` value to switch. Runs
  inside `asyncio.to_thread()` since the submodule uses sync Playwright.

All tool results cached in-process via `cachetools.TTLCache(maxsize=512, ttl=300)`.

## Storage

- **SQLite** (`./retail_catalog.db`) — products table with `structured_data` JSON column, queried via `json_extract` (SQLite JSON1)
- **BM25Okapi corpus** — rebuilt on startup from the SQLite catalog, kept in-memory

All state lives on the AMD GPU droplet's local NVMe. No external object store.

## Data flow

```
user message
   │
   ▼
DiscoveryAgent  ──► JSON brief (intent, brands, budget_max, must_have, …)
   │
   ▼
SynthesisAgent  ──► tool calls
                      │
                      ├── search_catalog  ──► BM25 ──► SQLite (local catalog)
                      │
                      └── fetch_product_from_site   ──► Amazon
                                                       (vendor/ecommerce-scraper submodule,
                                                        sync Playwright → asyncio.to_thread)
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