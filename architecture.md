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
| Discovery | Build a structured brief, record assumptions, and ask only blocking clarification questions | None |
| Catalog Research | Retrieve candidates, classify product identity, and label evidence gaps | `find_categories`, `search_catalog` |
| Critic | Enforce the brief, preserve eligible-product order, and propose contextual refinement chips | None |

Each agent is one MAF `Agent`. The Catalog Research agent uses schema-aware MAF tools; the framework validates each tool call, executes the read-only Python function, and returns its JSON evidence to the model.

## Interaction loop

The default path shows products without interruption:

- Discovery proceeds with reasonable, explicit assumptions for non-blocking ambiguity such as an unspecified brand, color, or budget.
- It asks one specific question only when compatibility is unknown, interpretations imply different product types, explicit constraints conflict, or a must-have would otherwise be silently relaxed.
- The orchestration layer allows at most two clarification questions and then requires the best supported brief.
- Explicit constraints are never silently relaxed. Conflicts are presented as choices, not generic confirmation prompts.
- Catalog searches and final ranking run automatically because they are read-only and cannot place orders or modify user data.

The Critic returns up to four contextual refinement chips derived from assumptions, evidence gaps, or useful trade-offs. Selecting a chip appends its instruction to the current request and reruns Discovery → Research → Critic. Free-text refinement follows the same path.

## Catalog

The external CSV dataset is imported once into `retail_catalog.db`. Neither the source CSV nor generated database is committed.

SQLite stores facts once:

```text
categories(id INTEGER PK, name UNIQUE)
products(id INTEGER PK, asin UNIQUE, ..., category_id FK -> categories.id)
product_fts(title, external content -> products.id)
```

FTS5 searches product titles and returns up to 50 candidates in BM25 order. SQL applies exact category, price, rating, and bestseller filters first. Research then classifies each title as `exact_product`, `accessory`, `unrelated`, or `uncertain`; only exact products remain eligible.

Eligible candidates are ordered deterministically by 55% retrieval relevance, 25% log-scaled `bought_last_month`, 15% rating confidence from stars and review count, and 5% bestseller status. Log scaling prevents a single high-volume item from dominating. Product identity is an eligibility rule, so a popular accessory can never outrank the requested product. SQLite and FTS5 remain the only search infrastructure.

## Data flow

```text
user request
    |
    v
Discovery Agent -- blocking ambiguity? --> user clarification (maximum two)
    |
    v
structured brief
    |
    v
Catalog Research Agent -- find_categories, search_catalog (top 50)
    |                                    SQLite FTS5 + indexed filters
    v                                    (1.4M+ product dataset)
LLM product-type screening -- keep exact products only
    |
    v
deterministic multi-field ranking + explicit evidence gaps
    |
    v
Critic Agent --> ranked recommendation + refinement chips
    ^                                      |
    |-------- selected refinement ---------|
```

The system never claims live availability or current pricing. Prices, ratings, review counts, bestseller flags, and popularity are dataset snapshots.

## Benchmark

`bench/run_agent_bench.py` runs five full collaborations and records:

- end-to-end latency per scenario
- clarification count per scenario
- refinement-chip count per scenario
- catalog-tool cache hits and misses
- candidate and recommendation counts
- exact-product and excluded-product counts
- AMD GPU snapshots when available
- vLLM prefix-cache/request metrics when available
