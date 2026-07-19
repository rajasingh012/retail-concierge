"""Microsoft Agent Framework tools over the offline SQLite catalog."""
from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

from agent_framework import tool

from infrastructure.database import ProductCatalogRepository

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
    """Return cache evidence for benchmark reports."""
    return {
        "hits": _CACHE_HITS,
        "misses": _CACHE_MISSES,
        "size": len(_TOOL_CACHE),
        "maxsize": _CACHE_MAXSIZE,
    }


def clear_cache() -> None:
    """Clear cached tool results and counters."""
    global _CACHE_HITS, _CACHE_MISSES
    _TOOL_CACHE.clear()
    _CACHE_HITS = 0
    _CACHE_MISSES = 0


def build_tools(repository: ProductCatalogRepository) -> list[Any]:
    """Bind catalog queries into schema-aware MAF tools."""

    @tool
    def find_categories(query: str, limit: int = 10) -> str:
        """Find catalog categories and their integer IDs by name.

        Args:
            query: Category name fragment such as "office" or "headphones".
            limit: Maximum category matches to return, from 1 to 50.

        Returns:
            JSON array with category_id, category_name, and product_count.
        """
        key = json.dumps(["find_categories", query, limit], ensure_ascii=False)
        return _cached(
            key,
            lambda: json.dumps(
                repository.find_categories(query, limit), ensure_ascii=False
            ),
        )

    @tool
    def search_catalog(
        query: str,
        category_id: int = 0,
        max_price: float = 0,
        min_stars: float = 0,
        bestseller_only: bool = False,
        limit: int = 10,
    ) -> str:
        """Search the offline Amazon catalog using title text and exact filters.

        Args:
            query: Discriminating title terms. Use fewer terms to broaden a search.
            category_id: Exact category ID from find_categories; 0 means any category.
            max_price: Maximum listed dataset price; 0 means no ceiling.
            min_stars: Minimum rating from 0 to 5; 0 means no minimum.
            bestseller_only: If true, return only products marked bestseller.
            limit: Maximum products to return, from 1 to 50.

        Returns:
            JSON product evidence. Prices and ratings are dataset snapshots, not live data.
        """
        key = json.dumps(
            [
                "search_catalog", query, category_id, max_price, min_stars,
                bestseller_only, limit,
            ],
            ensure_ascii=False,
        )
        return _cached(
            key,
            lambda: json.dumps(
                [
                    product.to_dict()
                    for product in repository.search(
                        query,
                        category_id=category_id,
                        max_price=max_price,
                        min_stars=min_stars,
                        bestseller_only=bestseller_only,
                        limit=limit,
                    )
                ],
                ensure_ascii=False,
            ),
        )

    return [find_categories, search_catalog]
