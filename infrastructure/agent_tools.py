"""Microsoft Agent Framework tools over the ABO SQLite catalog."""
from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

from agent_framework import tool

from infrastructure.database import ABOCatalogRepository

_CACHE_MAXSIZE = 512
_TOOL_CACHE: OrderedDict[str, str] = OrderedDict()
_CACHE_HITS = 0
_CACHE_MISSES = 0


def _cached(key: str, factory) -> str:
    global _CACHE_HITS, _CACHE_MISSES
    if key in _TOOL_CACHE:
        _CACHE_HITS += 1
        _TOOL_CACHE.move_to_end(key)
        return _TOOL_CACHE[key]
    _CACHE_MISSES += 1
    value = factory()
    _TOOL_CACHE[key] = value
    if len(_TOOL_CACHE) > _CACHE_MAXSIZE:
        _TOOL_CACHE.popitem(last=False)
    return value


def cache_stats() -> dict[str, int]:
    return {"hits": _CACHE_HITS, "misses": _CACHE_MISSES, "size": len(_TOOL_CACHE), "maxsize": _CACHE_MAXSIZE}


def clear_cache() -> None:
    global _CACHE_HITS, _CACHE_MISSES
    _TOOL_CACHE.clear()
    _CACHE_HITS = 0
    _CACHE_MISSES = 0


def build_tools(repository: ABOCatalogRepository) -> list[Any]:

    @tool
    def find_categories(query: str, limit: int = 10) -> str:
        """Find catalog product types by name.

        Args:
            query: Product-type name fragment such as "office" or "headphone".
            limit: Maximum matches, from 1 to 50.

        Returns:
            JSON array with product_type and product_count.
        """
        key = json.dumps(["find_categories", query, limit], ensure_ascii=False)
        return _cached(
            key,
            lambda: json.dumps(repository.find_product_types(query, limit), ensure_ascii=False),
        )

    @tool
    def search_catalog(
        query: str,
        product_type: str = "",
        max_dimension_cm: float = 0.0,
        limit: int = 50,
    ) -> str:
        """Search the ABO catalog using title BM25 and structured filters.

        Args:
            query: Discriminating title terms. Use fewer terms to broaden a search.
            product_type: Exact product-type string from find_categories; empty means any type.
            max_dimension_cm: Maximum length dimension in centimeters (chairs, desks, etc.); 0 means no ceiling.
            limit: Candidate-pool size, from 1 to 50; default 50 so the Research agent
                can classify broadly before deterministic ranking.

        Returns:
            JSON listing array. Each listing has item_id, title_en, brand_en,
            product_type, product_url, marketplace, country, and boolean flags
            has_bullet, has_dimensions, has_weight, has_material. Retrieve
            structured attributes (color, material, style, dimensions) via
            listing_text_values / listing_dimensions tables — not included here.
        """
        key = json.dumps(
            ["search_catalog", query, product_type, max_dimension_cm, limit],
            ensure_ascii=False,
        )
        return _cached(
            key,
            lambda: json.dumps(
                [
                    {**listing, "retrieval_rank": rank}
                    for rank, listing in enumerate(
                        repository.search(
                            query,
                            product_type=product_type,
                            max_dimension_cm=max_dimension_cm,
                            limit=limit,
                        ),
                        start=1,
                    )
                ],
                ensure_ascii=False,
            ),
        )

    return [find_categories, search_catalog]
