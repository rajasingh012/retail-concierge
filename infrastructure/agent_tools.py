"""MAF tool wrappers around infrastructure services.

Exposes the BM25 search engine and the E-Commerces-WebScraper adapter
(vendor/ecommerce-scraper submodule) as plain callables decorated with
`@tool` for automatic schema registration.

Tool results are cached in-process via cachetools.TTLCache so
multi-turn agent loops (Discovery → Synthesis) don't re-hit the
catalog or re-launch a Playwright page for identical inputs.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from agent_framework import tool
from cachetools import TTLCache

from infrastructure.ecommerce_adapter import ECommerceAdapter
from infrastructure.indexer import LocalHybridSearchEngine


# ---------- tool-result cache ----------
# Sized to a single retail session: ~5 turns × ~3 tool calls each.
# 5-minute TTL so stale prices/reviews eventually refresh.
_TOOL_CACHE: TTLCache = TTLCache(maxsize=512, ttl=300)


def _cache_key(namespace: str, *parts: Any) -> str:
    """Stable hash of (tool_name, args). Args must be JSON-serializable."""
    raw = json.dumps(parts, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{namespace}:{digest}"


def cache_stats() -> Dict[str, int]:
    """Exposed for the bench script to report cache effectiveness."""
    return {"size": len(_TOOL_CACHE), "maxsize": _TOOL_CACHE.maxsize}


def clear_cache() -> None:
    _TOOL_CACHE.clear()


def build_tools(
    search_engine: LocalHybridSearchEngine,
    ecommerce_adapter: ECommerceAdapter | None = None,
) -> List[Any]:
    """Bind infrastructure instances into MAF tool definitions.

    Args:
        search_engine: BM25 search engine (required — local catalog lookup).
        ecommerce_adapter: E-Commerces-WebScraper adapter (required for live
            product lookups — provides Amazon/AliExpress/Shein/Shopee/
            MercadoLivre scraping via the vendor submodule).
    """

    @tool
    async def search_catalog(query: str, limit: int = 10) -> str:
        """Search the local product catalog using exact-keyword BM25 ranking.

        Args:
            query: Natural-language query. Brand names like 'Steelcase' or
                'Herman Miller' match with high precision.
            limit: Maximum number of hits to return (default 10).

        Returns:
            JSON array of hits: [{"product_id": str, "title": str, "score": float}]
        """
        key = _cache_key("search_catalog", query, limit)
        cached = _TOOL_CACHE.get(key)
        if cached is not None:
            return cached
        hits = search_engine.search(query, limit=limit)
        payload = json.dumps(
            [
                {"product_id": h.product_id, "title": h.title, "score": h.score}
                for h in hits
            ],
            ensure_ascii=False,
        )
        _TOOL_CACHE[key] = payload
        return payload

    @tool
    async def fetch_product_from_site(url: str, platform: str = "") -> str:
        """Fetch a product page from a major ecommerce site using pre-configured selectors.

        Uses the E-Commerces-WebScraper (vendor/ecommerce-scraper submodule).
        Handles Amazon, AliExpress, Mercado Livre, Shein, and Shopee.
        No manual CSS selectors needed — the scraper knows each site's structure.

        Args:
            url: Full product URL (e.g. "https://amazon.com/dp/B0ABC123").
            platform: Site name ("amazon", "aliexpress", "mercadolivre", "shein",
                "shopee"). If empty, auto-detects from the URL.

        Returns:
            JSON with title, price, description, and metadata.
        """
        if ecommerce_adapter is None:
            return json.dumps({"error": "E-Commerces-WebScraper adapter not configured."})
        pl = platform if platform else None
        key = _cache_key("fetch_product_from_site", url, pl or "")
        cached = _TOOL_CACHE.get(key)
        if cached is not None:
            return cached
        result = await ecommerce_adapter.fetch_json(url, platform=pl)
        _TOOL_CACHE[key] = result
        return result

    tools = [search_catalog]
    if ecommerce_adapter is not None:
        tools.append(fetch_product_from_site)
    return tools
