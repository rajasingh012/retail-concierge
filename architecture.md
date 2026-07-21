# Architecture

## Layers

```text
domain/          (empty — agents use plain dicts from catalog)
use_cases/       Discovery, Catalog Research, and Critic agents
infrastructure/  SQLite catalog, FTS5 search, MAF tools, chat client factory
scripts/         ABO NDJSON importer, vLLM launcher, droplet firewall
bench/           collaboration benchmark and AMD metrics
main.py          composition root and interactive orchestration
```

Agents receive Microsoft Agent Framework's `OpenAIChatCompletionClient`; vLLM and DeepSeek use the same OpenAI Chat Completions wire protocol.

## Agent collaboration

| Agent | Responsibility | Tools |
|---|---|---|
| Discovery | Build a structured brief, record assumptions, and ask only blocking clarification questions | None |
| Catalog Research | Retrieve candidates, classify product identity, and label evidence gaps | `find_product_types`, `search_catalog` |
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

The Amazon Berkeley Objects (ABO) NDJSON dataset is imported once into `retail_catalog.db`. The raw shards stay outside Git.

SQLite stores facts once:

```text
listings(item_id TEXT PK, title_en TEXT, brand_en TEXT, product_type TEXT, ...)
listing_text_values(item_id FK, type, value)   — color, material, style, etc.
listing_dimensions(item_id FK, dimension, value, unit)  — height/width/length/weight in cm/g
listings_fts(title, brand, content -> listings)
```

FTS5 searches title + brand and returns up to 50 candidates in BM25 order. SQL applies optional product-type and dimension filters first. The Research agent then classifies each listing as `exact_product`, `accessory`, `unrelated`, or `uncertain`; only exact products remain eligible.

Eligible candidates are ordered deterministically by 55% FTS5 relevance + 25% bullet-coverage score + 15% material/brand presence + 5% dimension availability. No pricing, ratings, or popularity signals exist in the catalog.

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
Catalog Research Agent -- find_product_types, search_catalog (top 50)
    |                                    SQLite FTS5 + product-type/dimension filters
    v                                    (145K listings, 576 product types)
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

The system never claims live availability or current pricing. The catalog contains no prices, ratings, review counts, or popularity data.

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
