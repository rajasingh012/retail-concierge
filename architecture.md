# Architecture

## Layers

```text
domain/          catalog evidence contracts
use_cases/       one conversational shopping agent and deterministic ranking
infrastructure/  SQLite catalog, FTS5 search, MAF tools, chat client factory
scripts/         ABO NDJSON importer, vLLM launcher, droplet firewall
bench/           agent benchmark and AMD metrics
main.py          composition root and interactive conversation loop
```

The agent receives Microsoft Agent Framework's `OpenAIChatCompletionClient`; vLLM and DeepSeek use the same OpenAI Chat Completions wire protocol.

## Agent and tools

RetailConcierge is one MAF `Agent` responsible for the complete user conversation:

- ask only blocking clarification questions
- retain answers and refinements in one `AgentSession`
- call `find_product_types` and `search_catalog`
- classify every retrieved item as `exact_product`, `accessory`, `unrelated`, or `uncertain`
- explain supported recommendations and evidence gaps

The agent calls `finalize_recommendations` after classification. That application-owned tool removes every non-exact product and applies deterministic ranking. The final response is checked against that protected candidate set, so the model cannot restore an excluded product, introduce an unknown ID, or change the deterministic order.

## Conversation

```text
user message
    |
    v
RetailConcierge Agent (shared AgentSession)
    |-- blocking ambiguity --> concise question --> user answer --|
    |                                                        <-----|
    |-- find_product_types (when useful)
    |-- search_catalog (up to 50 BM25 candidates)
    |-- classify product identity
    |-- finalize_recommendations
    v
protected ranked products + evidence notes + refinement chips
    |
    `-- user follow-up or selected refinement --> same AgentSession
```

The default path shows products without interruption. Compatibility uncertainty, fundamentally different product interpretations, conflicting explicit constraints, or silent relaxation of a must-have can trigger one question. Missing budget, brand, color, or a nice-to-have does not block useful results.

## Catalog

The Amazon Berkeley Objects (ABO) NDJSON dataset is imported once into `retail_catalog.db`. Raw shards stay outside Git.

```text
listings(id INTEGER PK, item_id, title_en, brand_en, product_type, ...)
listing_text_values(id INTEGER PK, listing_id FK, attribute, value)
listing_dimensions(id INTEGER PK, listing_id FK, dimension, value, unit)
listing_fts(title_en, brand_en, content -> listings)
```

SQLite FTS5 searches title and brand and returns up to 50 candidates in BM25 order. SQL can apply exact product-type and dimension filters first. Product identity is then screened semantically by the agent; keyword blacklists are not used.

Each `search_catalog` call records the returned `item_id` values into the session's `CatalogEvidenceTracker`. The agent's `finalize_recommendations` tool drops any candidate whose `item_id` was never observed by `search_catalog` in the current session, so invented or hallucinated IDs cannot enter the displayed list. The candidate order produced by the finalizer is authoritative.

Eligible candidates are ordered deterministically by:

- 50% FTS5 relevance
- 15% bullet coverage
- 15% material presence
- 10% brand presence
- 10% dimension evidence

The catalog contains no prices, ratings, review counts, popularity data, or live availability. The system never claims those facts.

## Session boundary

The CLI creates one `AgentSession` and reuses it until the user exits. MAF stores the turn history in that session, allowing a clarification answer or refinement to continue the same conversation. The CLI does not persist sessions across process restarts.

## Benchmark

`bench/run_agent_bench.py` creates an independent session for each scenario and records:

- end-to-end latency
- clarification or recommendation response kind
- recommendation and refinement-chip counts
- catalog-tool cache hits and misses
- AMD GPU snapshots when available
- vLLM metrics when available
