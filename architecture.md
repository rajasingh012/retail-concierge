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
| Discovery | Build a structured brief from the user's request | `commit_brief` (approval-mode) |
| Catalog Research | Query only the offline catalog and label evidence gaps | `find_categories`, `search_catalog` (both approval-mode) |
| Critic | Reject unsupported matches and rank evidence against the brief | `submit_recommendation` (approval-mode) |

Each agent is one MAF `Agent`. Reasoning and tool use are interleaved inside the same agent; the framework handles session resume, message history, and stream buffering.

## Human-in-the-loop

The user is the gate. Every tool that selects evidence or commits a final answer runs through `ApprovalMiddleware`:

- `search_catalog` and `find_categories` pause so the user can approve, edit, or redirect the proposed query before it executes.
- `commit_brief` and `submit_recommendation` pause so the user can approve, edit, or reject the agent's structured output before it becomes the basis for the next stage.
- `find_categories` is auto-approved when the result is non-empty; the user only sees the gate when the agent needs to commit a category it could not verify.

The CLI reads one line per approval. The benchmark passes an `AutoApproveMiddleware` that replies to every approval with the original call, so the bench stays headless and reproducible.

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
Discovery Agent -- commit_brief (gated) --> structured brief
    |
    v
Catalog Research Agent -- find_categories, search_catalog (each gated)
    |                                       SQLite FTS5 + indexed filters
    v                                       (1.4M+ product dataset)
research evidence + explicit evidence gaps
    |
    v
Critic Agent -- submit_recommendation (gated) --> ranked recommendation
```

Every arrow that crosses an agent boundary is a tool call. Every tool call is a user checkpoint.

The system never claims live availability or current pricing. Prices, ratings, review counts, bestseller flags, and popularity are dataset snapshots.

## Benchmark

`bench/run_agent_bench.py` runs five full collaborations and records:

- end-to-end latency per scenario
- approval count (gates hit per scenario)
- catalog-tool cache hits and misses
- candidate and recommendation counts
- AMD GPU snapshots when available
- vLLM prefix-cache/request metrics when available
