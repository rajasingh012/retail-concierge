# Architecture

## Layers

```text
domain/          catalog evidence contracts
use_cases/       shopping agent, brief extraction, and deterministic ranking
infrastructure/  SQLite catalog, FTS5 search, MAF tools, chat client factory
scripts/         ABO NDJSON importer, vLLM launcher, droplet firewall
bench/           agent benchmark and AMD metrics
main.py          composition root and interactive conversation loop
```

The agent receives Microsoft Agent Framework's `OpenAIChatCompletionClient`; vLLM and DeepSeek use the same OpenAI Chat Completions wire protocol.

## Agent and tools

RetailConcierge is one MAF `Agent` responsible for the complete user conversation:

- call `extract_brief` first to produce a structured brief via LLM tool calling
- ask only blocking clarification questions (capped at 2 per turn by the brief tool)
- call `find_product_types` and `find_brands` to canonicalize names against the catalog
- call `search_catalog` to retrieve BM25 candidates
- classify every retrieved item as `exact_product`, `accessory`, `unrelated`, or `uncertain`
- call `finalize_recommendations` for the catalog-provenance gate and deterministic ranking
- explain supported recommendations and evidence gaps

The five MAF tools, in call order:

| Tool | Description |
|---|---|
| `extract_brief` | LLM args. Agent fills a `ShoppingBrief` Pydantic model (intent, search_terms, product_type, brand, budget_usd converted to USD, max_dimension_cm converted to cm, must_have, nice_to_have, color, material, compatibility, target_use, quantity, assumptions, evidence_gaps) via MAF tool calling. Tool body returns the validated brief dict. No database access, no offline parsing — the LLM owns field extraction. |
| `find_product_types` | LIKE-match against the `product_type` column, ordered by listing count. |
| `find_brands` | Three-tier brand resolution: exact prefix → FTS5 (stemming + close misspellings) → LIKE fallback. |
| `search_catalog` | BM25 via FTS5, up to 50 candidates with optional product-type, brand, and dimension filters. Records observed `item_id` values into the session's `CatalogEvidenceTracker`. |
| `finalize_recommendations` | Drops candidates whose `item_id` was not observed by `search_catalog` in the current session, keeps only `exact_product`, applies deterministic multi-field ranking, and returns the authoritative order. |

## Conversation

```text
                   ┌──────────────────────────────────────────────┐
                   │          user message / refinement           │
                   └────────────────────┬─────────────────────────┘
                                        │
                                        v
                   ┌──────────────────────────────────────────────┐
                   │           extract_brief                      │
                   │  (LLM fills ShoppingBrief via tool calling   │
                   │   — currency / dimension / quantity          │
                   │   conversion, assumptions, evidence gaps)    │
                   └─────────┬────────────────────┬───────────────┘
                             │                    │
                    complete=false          complete=true
                    question="..."          with brief
                             │                    │
                             v                    v
                  ┌────────────────────┐  ┌──────────────────────────┐
                  │ concise question   │  │ find_product_types /     │
                  │    → user answer   │  │ find_brands (canonicalize│
                  │    (max 2 per turn)│  │  against catalog)        │
                  └────────┬───────────┘  └────────────┬─────────────┘
                           │                            │
                           └──── (back to extract_brief)│
                                                        v
                                        ┌──────────────────────────────┐
                                        │       search_catalog         │
                                        │  (BM25, up to 50 candidates) │
                                        └────────────┬─────────────────┘
                                                     │
                                                     v
                                        ┌──────────────────────────────┐
                                        │  classify product identity   │
                                        │  → exact_product / accessory │
                                        │    / unrelated / uncertain   │
                                        └────────────┬─────────────────┘
                                                     │
                                                     v
                                        ┌──────────────────────────────┐
                                        │  finalize_recommendations    │
                                        │  • catalog-provenance gate   │
                                        │  • deterministic ranking     │
                                        └────────────┬─────────────────┘
                                                     │
                                                     v
                                        ┌──────────────────────────────┐
                                        │  protected ranked products   │
                                        │  + evidence notes           │
                                        │  + assumptions              │
                                        │  + refinement chips         │
                                        └────────────┬─────────────────┘
                                                     │
                                        user follow-up or selected chip
                                                     │
                                                     └──→ same AgentSession
                                                          (fresh brief)
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
