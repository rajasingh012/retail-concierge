# Architecture

## Layers

```mermaid
graph LR
  domain["domain/<br/>catalog evidence contracts"]
  use_cases["use_cases/<br/>shopping agent + ranking"]
  infra["infrastructure/<br/>SQLite FTS5 + MAF tools + chat clients"]
  scripts["scripts/<br/>importer + vLLM launcher + audit_verify"]
  bench["bench/<br/>agent benchmark + AMD metrics"]
  main["main.py<br/>composition root + CLI loop"]
  app["app.py<br/>Streamlit UI"]

  main --> infra
  main --> use_cases
  main --> domain
  app --> infra
  app --> use_cases
  app --> domain
  use_cases --> infra
  use_cases --> domain
  infra --> domain
  bench --> main
  scripts -.imports.-> infra
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

| Tool | What it does |
|---|---|
| `extract_brief` | LLM fills a `ShoppingBrief` Pydantic model; tool body validates (typed currency / dimension / quantity conversion, vocabulary gate). |
| `find_product_types` | LIKE-match against the `product_type` column, ordered by listing count. |
| `find_brands` | Three-tier resolution: exact prefix → FTS5 → LIKE fallback. Handles misspellings and case. |
| `search_catalog` | BM25 via FTS5, up to 50 candidates with optional type / brand / dimension filters. Records observed `item_id` values into the session's tracker. |
| `finalize_recommendations` | Drops candidates whose `item_id` was not seen by `search_catalog` in this session, keeps only `exact_product`, applies deterministic multi-field ranking. Returns the authoritative order. |

## Conversation

```mermaid
flowchart TD
  user(["user message / refinement"])
  brief["extract_brief<br/>(LLM fills ShoppingBrief)"]
  qcheck{brief complete?}
  question["concise question<br/>max 2 per turn"]
  resolve["find_product_types / find_brands<br/>(canonicalize against catalog)"]
  search["search_catalog<br/>(BM25 + filters)<br/>records item_ids to tracker"]
  classify["classify each item<br/>exact_product / accessory /<br/>unrelated / uncertain"]
  finalize["finalize_recommendations<br/>• provenance gate<br/>• deterministic ranking<br/>• audit-log entry"]
  out["protected ranked products<br/>+ evidence notes<br/>+ assumptions<br/>+ refinement chips<br/>+ audit-log entry"]

  user --> brief
  brief --> qcheck
  qcheck -- "complete=false" --> question
  qcheck -- "complete=true" --> resolve
  question --> user
  resolve --> search
  search --> classify
  classify --> finalize
  finalize --> out
  out --> user
```

The default path shows products without interruption. Compatibility uncertainty, fundamentally different product interpretations, conflicting explicit constraints, or silent relaxation of a must-have can trigger one question. Missing budget, brand, color, or a nice-to-have does not block useful results.

## Catalog

The Amazon Berkeley Objects (ABO) NDJSON dataset is imported once into `retail_catalog.db`. Raw shards stay outside Git.

```mermaid
erDiagram
  LISTINGS ||--o{ LISTING_TEXT_VALUES : "has"
  LISTINGS ||--o{ LISTING_DIMENSIONS : "has"
  LISTINGS ||--o| LISTING_FTS : "indexed by"

  LISTINGS {
    INTEGER id PK
    string  item_id
    string  title_en
    string  brand_en
    string  product_type
    string  product_url
    string  marketplace
    string  country
  }
  LISTING_TEXT_VALUES {
    INTEGER id PK
    INTEGER listing_id FK
    string  attribute
    string  value
  }
  LISTING_DIMENSIONS {
    INTEGER id PK
    INTEGER listing_id FK
    string  dimension
    real    value
    string  unit
  }
  LISTING_FTS {
    string    title_en
    string    brand_en
    INTEGER   content FK
  }
```

SQLite FTS5 searches title and brand and returns up to 50 candidates in BM25 order. SQL can apply exact product-type and dimension filters first. Product identity is then screened semantically by the agent; keyword blacklists are not used.

Each `search_catalog` call records the returned `item_id` values into the session's `CatalogEvidenceTracker`, and `finalize_recommendations` drops any candidate whose `item_id` was not seen in this session — so invented IDs cannot reach the displayed list. The finalizer's candidate order is authoritative.

Eligible candidates are ordered deterministically by:

- 50% FTS5 relevance
- 15% bullet coverage
- 15% material presence
- 10% brand presence
- 10% dimension evidence

The catalog contains no prices, ratings, review counts, popularity data, or live availability. The system never claims those facts.

## Misspelling, foreign-language, and paraphrase handling

`extract_brief` is the single point per turn where the LLM maps user words to canonical catalog values. It runs first, before any search tool call.

The system prompt appends the catalog's `product_type` and `brand` vocabularies. The LLM canonicalizes at write time: misspellings ("ofice chair"), foreign-language terms ("chaise de bureau", "krzesło biurowe"), paraphrases ("executive seating"), abbreviations ("ofc chr") all map to the closest catalog value. A brief-level Pydantic validator rejects off-vocabulary values so the model cannot silently pass wrong types/brands into `search_catalog`. This handles all four input variations in the one tool call we already make, instead of stacking database-side fuzzy indexes per field. Implementation lives in `domain/recommendation.py` (validator) and `use_cases/shopping_agent.py` (prompt composition); tests in `tests/test_brief_vocabulary_gate.py`.

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

## Audit log

Every catalog and finalizer tool call writes one entry to an append-only JSONL file. Each entry links to the previous one via `prev_hash`; the chain head is the most recent `entry_hash`; the chain is `sha256(canonical_json(entry_without_entry_hash))`. `extract_brief` is not recorded (no external-data semantics). The logger is opt-in: `RETAIL_AUDIT_LOG=./retail_audit.jsonl`.

```mermaid
flowchart LR
  G["genesis<br/>prev_hash = 0…0"]
  E1["entry 1<br/>prev_hash=0…0<br/>entry_hash=H1"]
  E2["entry 2<br/>prev_hash=H1<br/>entry_hash=H2"]
  E3["entry 3<br/>prev_hash=H2<br/>entry_hash=H3"]
  E4["entry 4<br/>prev_hash=H3<br/>entry_hash=H4"]
  V["audit_verify.py<br/>exit 0 = clean<br/>exit 1 = tampered"]

  G -. "seq=1 requires<br/>prev_hash=0…0" .-> E1
  E1 -- "prev_hash=H1<br/>must equal computed" --> V
  E2 -- "prev_hash=H2" --> V
  E3 -- "prev_hash=H3" --> V
  E4 -- "prev_hash=H4<br/>chain head" --> V
  E1 -- "entry_hash=H1" --> E2
  E2 -- "entry_hash=H2" --> E3
  E3 -- "entry_hash=H3" --> E4
```

```mermaid
flowchart LR
  proposed["model proposes to<br/>finalize_recommendations"]
  tracker["session tracker<br/>(item_ids from search_catalog)"]
  gate["provenance gate<br/>drop ∉ tracker"]
  accepted["accepted_item_ids<br/>(ranked, displayed)"]
  blocked["provenance_blocked<br/>(audit only,<br/>never displayed)"]

  proposed --> gate
  tracker --> gate
  gate -- "in tracker" --> accepted
  gate -- "not in tracker" --> blocked
```

`finalize_recommendations` records `proposed_item_ids`, `accepted_item_ids`, and `provenance_blocked` — the `provenance_blocked` field is the only artifact that proves the provenance gate had anything to catch.

Verify with `python scripts/audit_verify.py retail_audit.jsonl` (stdlib only, no project deps, works on the demo droplet without a venv). Implementation: `infrastructure/audit.py` (logger), `scripts/audit_verify.py` (verifier), tests `tests/test_audit_log.py`.
