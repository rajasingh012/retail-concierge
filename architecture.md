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
| Discovery | Build a structured brief and ask only decision-impacting clarification questions | None |
| Catalog Research | Query only the offline catalog and label evidence gaps | `find_categories`, `search_catalog` |
| Critic | Reject unsupported matches and rank evidence against the brief | None |

Each agent is one MAF `Agent`. The Catalog Research agent uses schema-aware MAF tools; the framework validates each tool call, executes the read-only Python function, and returns its JSON evidence to the model.

## Clarification loop

The user enters the loop only when a missing preference would materially change the recommendation:

- Discovery asks one specific question at a time, prioritizing intended use, hard constraints, and budget.
- The orchestration layer allows at most two clarification questions and then requires the best supported brief.
- Explicit constraints are never silently relaxed. Conflicts are presented as choices, not generic confirmation prompts.
- Catalog searches and final ranking run automatically because they are read-only and cannot place orders or modify user data.

The user can refine the recommendation in a later conversational turn. Catalog tools and recommendations are not user-confirmation checkpoints because the application does not add items to a cart or make purchases.

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
Discovery Agent -- material ambiguity? --> user clarification (maximum two)
    |
    v
structured brief
    |
    v
Catalog Research Agent -- find_categories, search_catalog
    |                                   SQLite FTS5 + indexed filters
    v                                   (1.4M+ product dataset)
research evidence + explicit evidence gaps
    |
    v
Critic Agent --> ranked recommendation
```

The system never claims live availability or current pricing. Prices, ratings, review counts, bestseller flags, and popularity are dataset snapshots.

## Benchmark

`bench/run_agent_bench.py` runs five full collaborations and records:

- end-to-end latency per scenario
- clarification count per scenario
- catalog-tool cache hits and misses
- candidate and recommendation counts
- AMD GPU snapshots when available
- vLLM prefix-cache/request metrics when available
