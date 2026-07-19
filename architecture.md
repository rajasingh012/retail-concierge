# Architecture

## Layers

```text
domain/          catalog evidence contracts
use_cases/       Discovery, Catalog Research, and Critic agents
infrastructure/  SQLite catalog, FTS5 search, MAF tools, chat client factory
scripts/         dataset importer, vLLM launcher, droplet firewall
bench/           collaboration benchmark and AMD metrics
main.py          composition root and interactive orchestration
```

`domain/` has no framework or infrastructure imports. Agents receive Microsoft Agent Framework's `OpenAIChatClient`; vLLM and DeepSeek use the same OpenAI-compatible client.

## Agent collaboration

| Agent | Responsibility | Tools |
|---|---|---|
| Discovery | Ask up to two decision-impacting questions and produce a structured brief | none |
| Catalog Research | Query only the offline catalog and label evidence gaps | `find_categories`, `search_catalog` |
| Critic | Reject unsupported matches and rank evidence against the brief | none |

Agent handoffs are explicit JSON objects. The orchestrator parses each handoff before passing it to the next agent.

## Catalog

The external CSV dataset is imported once into `retail_catalog.db`. Neither the source CSV nor generated database is committed.

SQLite stores facts once:

```text
categories(id INTEGER PK, name UNIQUE)
products(id INTEGER PK, asin UNIQUE, ..., category_id FK -> categories.id)
product_fts(title, external content -> products.id)
```

FTS5 searches product titles. SQL applies exact category, price, rating, bestseller, and result-limit filters before evidence enters model context. Products with zero dataset price are excluded from recommendations.

## Data flow

```text
user request
    |
    v
Discovery Agent -- optional questions --> structured brief
    |
    v
Catalog Research Agent -- tools --> SQLite FTS5 + indexed filters
    |                              (1.4M+ product dataset)
    v
research evidence + explicit evidence gaps
    |
    v
Critic Agent --> ranked recommendation + dataset limitations
```

The system never claims live availability or current pricing. Prices, ratings, review counts, bestseller flags, and popularity are dataset snapshots.

## Benchmark

`bench/run_agent_bench.py` runs five full collaborations and records:

- end-to-end latency per scenario
- clarification count
- candidate and recommendation counts
- catalog-tool cache hits and misses
- AMD GPU snapshots when available
- vLLM prefix-cache/request metrics when available
