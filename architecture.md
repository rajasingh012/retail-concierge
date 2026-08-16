# Architecture

## Layers

```mermaid
graph LR
  domain["domain/<br/>catalog evidence contracts + IntroBullet schema"]
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
- call `search_catalog` to retrieve BM25 candidates, writing observed `item_id`s into `ctx.session.state`
- classify every retrieved item as `exact_product`, `accessory`, `unrelated`, or `uncertain`
- call `finalize_recommendations` to drop anything not seen by `search_catalog` this session, keep only `exact_product`, apply deterministic weighted ranking with the intent-match tie-breaker, and emit the typed `IntroBullet` recommendation list
- after `finalize_recommendations`, narrate supported picks and evidence gaps; the bullet list itself is the ground truth

The five MAF tools, in call order:

| Tool | What it does |
|---|---|
| `extract_brief` | LLM fills a `ShoppingBrief` Pydantic model; tool body validates (typed currency / dimension / quantity conversion, vocabulary gate). Writes `target_use` and `must_have` into `ctx.session.state`. |
| `find_product_types` | LIKE-match against the `product_type` column, ordered by listing count. |
| `find_brands` | Three-tier resolution: exact prefix → FTS5 → LIKE fallback. Handles misspellings and case. |
| `search_catalog` | BM25 via FTS5, up to 50 candidates with optional type / brand / dimension filters. Writes returned `item_id` values into `ctx.session.state['seen_item_ids']`. |
| `finalize_recommendations` | Reads `seen_item_ids` / `target_use` / `must_have` from `ctx.session.state`. Drops candidates whose `item_id` was not seen by `search_catalog` in this session, keeps only `exact_product`, applies deterministic multi-field ranking with the intent-match tie-breaker, and returns the typed result with its `IntroBullet` recommendation list. |

Per-shopper memory lives entirely on the MAF `AgentSession` object — `seen_item_ids`, `target_use`, `must_have`. The CLI creates one `AgentSession` and reuses it across turns; Streamlit's "New Session" button creates a fresh one. There is no process-wide or closure-captured state.

## Conversation

```mermaid
flowchart TD
  user(["user message / refinement"])
  brief["extract_brief<br/>(LLM fills ShoppingBrief<br/>+ writes target_use, must_have to session.state)"]
  qcheck{brief complete?}
  question["concise question<br/>max 2 per turn"]
  resolve["find_product_types / find_brands<br/>(canonicalize against catalog)"]
  search["search_catalog<br/>(BM25 + filters)<br/>writes item_ids to session.state"]
  classify["classify each item<br/>exact_product / accessory /<br/>unrelated / uncertain"]
  finalize["finalize_recommendations<br/>• provenance gate (drop ∉ session.state)<br/>• deterministic ranking<br/>• intent-match tie-breaker<br/>• IntroBullet schema validation<br/>• audit-log entry"]
  out["protected ranked products<br/>+ typed IntroBullet list<br/>+ evidence notes<br/>+ assumptions<br/>+ refinement chips<br/>+ audit-log entry"]

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

## Recommendation output contract

The bullet list the shopper sees is a typed list of `IntroBullet` (Pydantic model in `domain/recommendation.py`), not free-form prose. Two closed enums lock the schema:

- `subject`: one of `brief`, `item`, `catalog`, `dataset`.
- `claim_kind`: a closed set bounded by what the catalog can evidence
  (`price`, `stock`, `shipping`, `rating`, `warranty`, `discount`) plus
  structural kinds (`intent_match`, `dataset_disclaimer`, `none`, …).
  No ad-hoc claim category can be emitted.

A pair of cross-field validators enforces invariants:

- `subject='item'` must carry an `item_id`; `subject='brief'` must not.
- `claim_kind='dataset_disclaimer'` must be about the dataset itself.
- `claim_kind='intent_match'` must reference `target_use` or `must_have`.

On top of validation, the finalizer applies three runtime guards before the response is returned:

1. A synthetic `dataset_disclaimer` bullet is appended so shoppers
   see "this is a dataset snapshot, not a live storefront."
2. Phantom-item bullets — items the model proposed but that were not
   in `search_catalog` output — are stripped. The provenance gate in
   `finalize_recommendations` already keeps them out of the ranked
   list; this guard keeps them out of the prose bullets too.
3. Any `CATALOG_NOTICE` overwrite is re-applied after the guard pass,
   so a finalizer cannot quietly remove the catalog-disclosure line.

Legacy string bullets are coerced into `IntroBullet` via a `BeforeValidator`, so older tool payloads still validate without code changes elsewhere. Implementation: `domain/recommendation.py` (schema + validators), `use_cases/shopping_agent.py` (runtime guards).

### Intent-match tie-breaker

`screen_and_rank_candidates` ranks by a deterministic weighted score (price / rating / stock / brand / type match). When two candidates tie on that primary score, `_target_use_match_score` supplies a secondary key that prefers the candidate whose `target_use` overlaps the shopper's brief. The primary score is never overridden — the tie-breaker only resolves draws, so the existing ranking contracts stay intact.

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

FTS5 returns BM25-ordered candidates with optional SQL filters for product type and dimension. `search_catalog` records returned `item_id`s into the session's `ctx.session.state`; `finalize_recommendations` reads that state and drops anything not seen — invented IDs cannot reach the displayed list or the bullet list. The catalog carries no prices, ratings, popularity, or availability; the IntroBullet `claim_kind` enum reflects this. Implementation: `infrastructure/database.py`, `use_cases/ranking.py`.

## Misspelling, foreign-language, and paraphrase handling

`extract_brief` is the single point per turn where the LLM maps user words to canonical catalog values. Its system prompt includes the catalog's `product_type` and `brand` vocabularies, and the LLM is instructed to canonicalize misspellings, foreign-language input, and paraphrases against that vocabulary. A brief-level Pydantic validator rejects off-vocabulary values so wrong types / brands cannot silently reach `search_catalog`. One tool call handles all four input variations, instead of stacking database-side fuzzy indexes per field. Implementation: `domain/recommendation.py` (validator), `use_cases/shopping_agent.py` (prompt composition); tests in `tests/test_brief_vocabulary_gate.py`.

## Session boundary

The CLI creates one `AgentSession` and reuses it until the user exits. MAF stores the turn history in that session, allowing a clarification answer or refinement to continue the same conversation. Between turns of the same conversation, `seen_item_ids` / `target_use` / `must_have` are reset on `ctx.session.state` so each new user message starts with a clean provenance scope. The CLI does not persist sessions across process restarts. Streamlit uses the same `AgentSession` object per active chat; "New Session" creates a fresh one.

## Benchmark

`bench/run_agent_bench.py` creates an independent session per scenario and emits a standardized record set (latency, response kind, recommendation / chip counts, catalog cache hits / misses, AMD GPU / vLLM metrics where available). Implementation: `bench/run_agent_bench.py`.

## Audit log

A judge, regulator, or customer can ask: *"What did the concierge actually access in this session, and can a third party prove it without trusting our codebase?"* The audit log answers both questions: every catalog or finalizer tool call writes one entry to an append-only JSONL file, and the entries form a tamper-evident hash chain verifiable by a stdlib-only script.

### Hash chain (one entry per tool call)

Each entry links to the previous via `prev_hash`/`entry_hash`. Edit any line, delete any line, or reorder any line — and `audit_verify.py` exits 1 with the offending line number.

```mermaid
flowchart LR
  G["entry 1<br/>prev_hash = 0…0<br/>(genesis)"]
  E1["entry 2<br/>prev_hash = H1"]
  E2["entry 3<br/>prev_hash = H2"]
  E3["entry 4<br/>prev_hash = H3"]
  V["audit_verify.py<br/>exit 0 = clean<br/>exit 1 = tampered"]

  G -- "entry_hash = H1" --> E1
  E1 -- "entry_hash = H2" --> E2
  E2 -- "entry_hash = H3" --> E3
  E3 -- "chain head" --> V
  V -- "re-hash every entry<br/>confirm prev_hash links" --> G
  V -- "re-hash every entry<br/>confirm prev_hash links" --> E1
  V -- "re-hash every entry<br/>confirm prev_hash links" --> E2
  V -- "re-hash every entry<br/>confirm prev_hash links" --> E3
```

### Provenance gate artifact (what gets logged per finalize call)

When `finalize_recommendations` runs, the log records the full picture: every item_id the model proposed, every item_id that survived the gate, and — crucially — every item_id the model tried to slip in that wasn't actually returned by `search_catalog` in this session. A clean log has empty `provenance_blocked`; a populated one is the audit story. The "seen this session" set is read from `ctx.session.state['seen_item_ids']`.

```mermaid
flowchart LR
  proposed["model proposes<br/>to finalize_recommendations"]
  state["ctx.session.state<br/>(item_ids from search_catalog)"]
  gate["provenance gate<br/>drop ∉ session.state"]
  accepted["accepted_item_ids<br/>displayed to user"]
  blocked["provenance_blocked<br/>audit-only,<br/>never displayed"]

  proposed --> gate
  state --> gate
  gate -- "in session.state" --> accepted
  gate -- "not in session.state" --> blocked
```

Opt-in via `RETAIL_AUDIT_LOG=./retail_audit.jsonl`. Verify with `python scripts/audit_verify.py retail_audit.jsonl` (stdlib only, works on the demo droplet without a venv). Implementation: `infrastructure/audit.py`, `scripts/audit_verify.py`; tests in `tests/test_audit_log.py`.
