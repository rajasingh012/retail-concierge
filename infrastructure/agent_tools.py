"""Microsoft Agent Framework tools over the ABO SQLite catalog."""
from __future__ import annotations

import json
from collections import OrderedDict
from typing import Annotated, Any

from pydantic import Field

from agent_framework import FunctionInvocationContext, tool

from infrastructure.database import ABOCatalogRepository

_CATALOG_NAMESPACE = "default"
_CACHE_MAXSIZE = 512


def use_cache_namespace(namespace: str) -> None:
    """Scope all subsequent tool caches to a different namespace."""
    global _CATALOG_NAMESPACE
    _CATALOG_NAMESPACE = namespace


_TOOL_CACHES: dict[str, "OrderedDict[str, str]"] = {}
_CACHE_STATS: dict[str, dict[str, int]] = {}


def _cache_state(namespace: str) -> tuple[OrderedDict[str, str], dict[str, int]]:
    cache = _TOOL_CACHES.setdefault(namespace, OrderedDict())
    stats = _CACHE_STATS.setdefault(namespace, {"hits": 0, "misses": 0, "size": 0, "maxsize": _CACHE_MAXSIZE})
    stats["maxsize"] = _CACHE_MAXSIZE
    return cache, stats


def _cached(key: str, factory) -> str:
    cache, stats = _cache_state(_CATALOG_NAMESPACE)
    if key in cache:
        stats["hits"] += 1
        cache.move_to_end(key)
        return cache[key]
    stats["misses"] += 1
    value = factory()
    cache[key] = value
    stats["size"] = len(cache)
    if len(cache) > _CACHE_MAXSIZE:
        cache.popitem(last=False)
        stats["size"] = len(cache)
    return value


def cache_stats() -> dict[str, int]:
    return dict(_CACHE_STATS.get(_CATALOG_NAMESPACE, {"hits": 0, "misses": 0, "size": 0, "maxsize": _CACHE_MAXSIZE}))


def clear_cache() -> None:
    cache, stats = _cache_state(_CATALOG_NAMESPACE)
    cache.clear()
    stats["hits"] = 0
    stats["misses"] = 0
    stats["size"] = 0


def clamp_limit(limit: int, *, default: int, maximum: int) -> int:
    """Clamp an LLM-supplied result count to a safe positive range."""
    try:
        parsed = int(limit)
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(parsed or default, maximum))


def build_tools(
    repository: ABOCatalogRepository,
    *,
    audit_logger: Any = None,
) -> list[Any]:
    """Build MAF tools over the catalog.

    Args:
        repository: ABO catalog repository.
        audit_logger: Optional ``AuditLogger`` instance. When set, every
            tool invocation writes one hash-chained JSONL entry.
    """
    db_path = str(repository._conn)
    repo_namespace = f"{_CATALOG_NAMESPACE}::{db_path}"

    def _record_observed(ctx: FunctionInvocationContext, candidates: list[dict]) -> None:
        if ctx.session is None:
            return
        seen = ctx.session.state.setdefault("seen_item_ids", set())
        for item in candidates:
            if not isinstance(item, dict):
                continue
            item_id = item.get("item_id")
            if item_id:
                seen.add(str(item_id))

    def _audit(tool: str, args: dict, result_meta: dict) -> None:
        if audit_logger is None:
            return
        record = getattr(audit_logger, "record", None)
        if record is None:
            return
        record(tool, args, result_meta)

    @tool(
        name="find_product_types",
        description=(
            "Find exact ABO catalog product_type values matching a short name "
            "fragment. Use only when a type filter will materially narrow search."
        ),
    )
    def find_product_types(
        query: Annotated[
            str,
            Field(description="Product-type fragment such as office or headphone"),
        ],
        limit: Annotated[
            int,
            Field(description="Maximum product types to return; clamped to 1-50"),
        ] = 10,
    ) -> str:
        safe_limit = clamp_limit(limit, default=10, maximum=50)
        key = json.dumps(
            [repo_namespace, "find_product_types", query, safe_limit],
            ensure_ascii=False,
        )
        types = repository.find_product_types(query, safe_limit)
        _audit(
            "find_product_types",
            {"query": query, "limit": safe_limit},
            {"result_count": len(types)},
        )
        return _cached(
            key,
            lambda: json.dumps(types, ensure_ascii=False),
        )

    @tool(
        name="find_brands",
        description=(
            "Find brand names in the ABO catalog matching a name fragment. "
            "Call this before search_catalog to identify the canonical brand "
            "name to filter by."
        ),
    )
    def find_brands(
        query: Annotated[
            str,
            Field(description="Brand name fragment such as ergocomfort or secretlab"),
        ],
        limit: Annotated[
            int,
            Field(description="Maximum brands to return; clamped to 1-20"),
        ] = 5,
    ) -> str:
        safe_limit = clamp_limit(limit, default=5, maximum=20)
        key = json.dumps(
            [repo_namespace, "find_brands", query, safe_limit],
            ensure_ascii=False,
        )
        brands = repository.find_brands(query, safe_limit)
        _audit(
            "find_brands",
            {"query": query, "limit": safe_limit},
            {"result_count": len(brands)},
        )
        return _cached(
            key,
            lambda: json.dumps(brands, ensure_ascii=False),
        )

    @tool(
        name="search_catalog",
        description=(
            "Search the offline ABO catalog with BM25 title relevance and optional "
            "exact product-type and maximum-dimension filters."
        ),
    )
    def search_catalog(
        ctx: Annotated[FunctionInvocationContext, "MAF context (excluded from schema)"],
        query: Annotated[
            str,
            Field(description="Concrete title terms; use fewer terms to broaden"),
        ],
        product_type: Annotated[
            str,
            Field(description="Exact value returned by find_product_types or empty"),
        ] = "",
        max_dimension_cm: Annotated[
            float,
            Field(description="Maximum length dimension in centimeters; 0 disables"),
        ] = 0.0,
        limit: Annotated[
            int,
            Field(description="Candidate-pool size; clamped to 1-50"),
        ] = 50,
    ) -> str:
        """Search the ABO catalog using title BM25 and structured filters.

        Args:
            query: Discriminating title terms. Use fewer terms to broaden a search.
            product_type: Exact product-type string from find_product_types; empty means any type.
            max_dimension_cm: Maximum length dimension in centimeters (chairs, desks, etc.); 0 means no ceiling.
            limit: Candidate-pool size, from 1 to 50; default 50 so the agent
                can classify broadly before deterministic ranking.

        Returns:
            JSON listing array. Each listing has item_id, title_en, brand_en,
            product_type, product_url, marketplace, country, and boolean flags
            has_bullet, has_dimensions, has_weight, has_material. Retrieve
            structured attributes (color, material, style, dimensions) via
            listing_text_values / listing_dimensions tables — not included here.
        """
        safe_limit = clamp_limit(limit, default=50, maximum=50)
        key = json.dumps(
            [
                repo_namespace,
                "search_catalog",
                query,
                product_type,
                max_dimension_cm,
                safe_limit,
            ],
            ensure_ascii=False,
        )
        candidates = [
            {**listing, "retrieval_rank": rank}
            for rank, listing in enumerate(
                repository.search(
                    query,
                    product_type=product_type,
                    max_dimension_cm=max_dimension_cm,
                    limit=safe_limit,
                ),
                start=1,
            )
        ]
        _record_observed(ctx, candidates)
        item_ids = [
            c.get("item_id") for c in candidates if isinstance(c, dict) and c.get("item_id")
        ]
        _audit(
            "search_catalog",
            {
                "query": query,
                "product_type": product_type,
                "max_dimension_cm": max_dimension_cm,
                "limit": safe_limit,
            },
            {"result_count": len(candidates), "item_ids": item_ids},
        )
        return _cached(
            key,
            lambda: json.dumps(candidates, ensure_ascii=False),
        )

    @tool(
        name="search_vector",
        description=(
            "Semantic KNN search over the catalog using sqlite-vec. Use as a "
            "complement to search_catalog when the user's query is phrased in "
            "natural language, contains synonyms, paraphrases, or misspellings "
            "that BM25 keyword matching would miss (e.g. 'couch' for 'sofa', "
            "'earphones' for 'headphones'). Returns up to 50 candidates ranked "
            "by cosine distance to the query embedding."
        ),
    )
    def search_vector(
        ctx: Annotated[FunctionInvocationContext, "MAF context (excluded from schema)"],
        query: Annotated[
            str,
            Field(
                description=(
                    "Natural-language search text — typically the same "
                    "search_terms you would pass to search_catalog, plus any "
                    "key color/material words. Encoded with MiniLM-L6 and "
                    "matched against the catalog via sqlite-vec KNN."
                )
            ),
        ],
        product_type: Annotated[
            str,
            Field(description="Exact value returned by find_product_types or empty"),
        ] = "",
        limit: Annotated[
            int,
            Field(description="Candidate-pool size; clamped to 1-50"),
        ] = 50,
    ) -> str:
        """Semantic KNN search over the catalog via the sqlite-vec extension.

        The query string is encoded with the same MiniLM-L6 model that built
        the index at ``scripts/build_vector_index.py`` time. Candidates are
        ranked by cosine distance; lower is better.

        Returned ``item_id`` values are written into
        ``ctx.session.state['seen_item_ids']`` exactly like ``search_catalog``
        does, so the provenance gate in ``finalize_recommendations`` works
        unchanged. Each candidate dict carries ``retrieval_backend='vector'``
        so callers can tell which path produced it (useful for the bench and
        for debugging).
        """
        safe_limit = clamp_limit(limit, default=50, maximum=50)
        key = json.dumps(
            [repo_namespace, "search_vector", query, product_type, safe_limit],
            ensure_ascii=False,
        )
        candidates: list[dict] = []
        try:
            query_vec = repository.encode_query(query)
            hits = repository.search_vector(
                query_vec, limit=safe_limit, product_type=product_type
            )
            candidates = hits
        except RuntimeError as exc:
            # sqlite-vec not loaded or vec_items not populated. Return a clear
            # empty result so the agent falls back to search_catalog instead
            # of crashing the tool loop.
            return json.dumps(
                {"error": str(exc), "candidates": []}, ensure_ascii=False
            )

        _record_observed(ctx, candidates)
        item_ids = [
            c.get("item_id") for c in candidates if isinstance(c, dict) and c.get("item_id")
        ]
        _audit(
            "search_vector",
            {"query": query, "product_type": product_type, "limit": safe_limit},
            {"result_count": len(candidates), "item_ids": item_ids},
        )
        return _cached(
            key,
            lambda: json.dumps(candidates, ensure_ascii=False),
        )

    return [find_product_types, find_brands, search_catalog, search_vector]
