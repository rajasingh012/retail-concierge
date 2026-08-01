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

| Tool | Description |
|---|---|
| `extract_brief` | LLM args. Agent fills a `ShoppingBrief` Pydantic model (intent, search_terms, product_type, brand, budget_usd converted to USD, max_dimension_cm converted to cm, must_have, nice_to_have, color, material, compatibility, target_use, quantity, assumptions, evidence_gaps) via MAF tool calling. Tool body returns the validated brief dict. No database access, no offline parsing — the LLM owns field extraction. |
| `find_product_types` | LIKE-match against the `product_type` column, ordered by listing count. |
| `find_brands` | Three-tier brand resolution: exact prefix → FTS5 (stemming + close misspellings) → LIKE fallback. |
| `search_catalog` | BM25 via FTS5, up to 50 candidates with optional product-type, brand, and dimension filters. Records observed `item_id` values into the session's `CatalogEvidenceTracker`. |
| `finalize_recommendations` | Drops candidates whose `item_id` was not observed by `search_catalog` in the current session, keeps only `exact_product`, applies deterministic multi-field ranking, and returns the authoritative order. |

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

Each `search_catalog` call records the returned `item_id` values into the session's `CatalogEvidenceTracker`. The agent's `finalize_recommendations` tool drops any candidate whose `item_id` was never observed by `search_catalog` in the current session, so invented or hallucinated IDs cannot enter the displayed list. The candidate order produced by the finalizer is authoritative.

Eligible candidates are ordered deterministically by:

- 50% FTS5 relevance
- 15% bullet coverage
- 15% material presence
- 10% brand presence
- 10% dimension evidence

The catalog contains no prices, ratings, review counts, popularity data, or live availability. The system never claims those facts.

## Brief-time vocabulary canonicalization

`extract_brief` is the single point per turn where the LLM maps the user's words to canonical catalog values. It runs first, before any search tool call.

The agent is given the catalog's vocabulary in the system prompt as two annotated lists:

- `CATALOG_PRODUCT_TYPES` — up to 80 canonical values, filtered by `HAVING COUNT(*) >= 5` so single-listing accidentals from import don't pollute the list.
- `CATALOG_BRANDS` — up to 200 canonical values, ordered by listing count, so the most-popular brands float to the top of the context window.

Instructions in the prompt require the LLM to canonicalize at write time: a misspelling ("ofice chair"), a foreign-language term ("chaise de bureau", "krzesło biurowe"), a paraphrase ("executive seating"), or an abbreviation ("ofc chr") must all map to the closest catalog value. `search_terms` carries the literal user terms first and a normalized form second, so FTS5 has both.

```mermaid
flowchart LR
  user(["user: 'ofice chair for <br/>studying at home'"])
  prompt["system prompt<br/>+ CATALOG_PRODUCT_TYPES list<br/>+ CATALOG_BRANDS list<br/>+ canonicalization rules"]
  brief["extract_brief<br/>(LLM returns ShoppingBrief)"]
  gate["Pydantic model_validator<br/>product_type, brand<br/>vs. seeded vocab"]
  search["search_catalog<br/>(product_type=CHAIR<br/>terms='office chair')"]

  user --> prompt
  prompt --> brief
  brief --> gate
  gate -- exact match --> search
  gate -- off-vocab --> err["ValueError to MAF<br/>(model retries<br/>with correct value)"]
  err --> brief
```

The validator is `domain.recommendation.ShoppingBrief._gate_against_catalog_vocabulary`. It is opt-in: the catalog calls `set_catalog_vocabulary(types, brands)` at agent build time, and the validator only fires when the seeded sets are non-empty. Empty values pass through (user did not specify), case variations fold to the catalog's canonical casing, and exact-match failures raise `ValidationError` which MAF surfaces to the model as a tool error. The model then retries with the correct value or omits the field, falling back to catalog-search-time discovery.

This handles all four misspelling classes at once:

1. Typo ("logtec") — canonical form folds via case-insensitive membership.
2. Foreign-language ("chaise de bureau") — LLM-native fuzzy mapping.
3. Paraphrase ("executive seating") — LLM-native fuzzy mapping.
4. Abbreviation ("ofc chr") — LLM-native fuzzy mapping.

A database-side fuzzy match would handle class 1 only. Adding a fuzzy index would not handle classes 2-4 at any reasonable cost; the LLM does it for free in the tool call we were already making.

Test coverage (`tests/test_brief_vocabulary_gate.py`, 9 tests):

- validator is a no-op when no vocab is seeded
- exact match passes
- case folds to the catalog's canonical casing
- unknown product_type / brand rejected with helpful error
- empty values bypass the gate
- reseeding changes behavior mid-process
- substring matching is strict (`headphone` ≠ `HEADPHONES`)
- whitespace-only entries in the seeded vocab are filtered out

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

Every catalog and finalizer tool call writes one entry to an append-only JSONL file. Each line is JSON with `seq`, `ts`, `session_id`, `tool`, `args`, `result_meta`, `prev_hash`, and `entry_hash` where `entry_hash = sha256(canonical_json(entry_without_entry_hash))`. Each entry's `prev_hash` links to the previous entry's `entry_hash`; the chain head is the most recent `entry_hash`. Genesis is 64 zeros. `canonical_json` is deterministic (sorted keys, UTF-8, no ASCII escapes, NaN/Inf rejected) so equal entries always hash equal.

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

Four tools are recorded: `find_product_types`, `find_brands`, `search_catalog`, and `finalize_recommendations`. `extract_brief` is not recorded — it has no external-data semantics. Each call uses a single write with `fsync`, so the file on disk always reflects the last completed entry. The logger holds a process-level lock around writes and refuses to reopen over a corrupt tail.

The logger is opt-in. Set `RETAIL_AUDIT_LOG=./retail_audit.jsonl` (or any path) to enable; leave unset for no behavior change. The `main.py` and `app.py` composition roots construct an `infrastructure.audit.AuditLogger` only when the environment variable is present, and pass it through `build_tools` and `build_shopping_agent`.

`finalize_recommendations` records three fields that matter for the audit story:

- `proposed_item_ids` — every item_id the model fed to the tool this call.
- `accepted_item_ids` — what survived the provenance gate and ranking, in authoritative order.
- `provenance_blocked` — `proposed - tracker.snapshot()`, i.e. item_ids the model tried to put on the list that were not actually returned by `search_catalog` in this session. Empty list means the model stayed within the catalog.

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

A separate `provenance_blocked` field is the only artifact that proves the gate had anything to catch, and the hash chain is what proves the log itself was not edited afterwards.

Verify with `python scripts/audit_verify.py retail_audit.jsonl` (stdlib only, no project deps, works on the demo droplet without a venv). Exit 0 means the chain is intact; exit 1 means a violation was found (line edit, line delete, reorder, or seq skip). `--summary` emits a machine-readable JSON report.
