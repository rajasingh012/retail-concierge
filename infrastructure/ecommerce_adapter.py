"""Adapter that wraps the E-Commerces-WebScraper repo (git submodule at vendor/)
into our ProductPayload domain entities.

This avoids reinventing Playwright selectors for Amazon, AliExpress, Shein,
Shopee, and Mercado Livre. Their scrapers are synchronous (sync_playwright);
we offload them to a thread via asyncio.to_thread() to stay async-safe.

Usage from agent tools:
    adapter = ECommerceAdapter()
    payload = await adapter.fetch_product("https://amazon.com/dp/...")
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from domain.entities import ItemVariant, ProductPayload


# Path to the submodule — importable after adding vendor/ to sys.path
_VENDOR_PATH = str(Path(__file__).resolve().parents[1] / "vendor" / "ecommerce-scraper")


# Supported platforms and their display names
PLATFORMS = {
    "amazon": "Amazon",
    "aliexpress": "AliExpress",
    "mercadolivre": "Mercado Livre",
    "shein": "Shein",
    "shopee": "Shopee",
}


def _import_scraper(platform: str) -> Any:
    """Lazy-import the correct scraper class from the submodule.

    The vendor directory is added to sys.path at import time so the
    scraper's internal imports (Logger, product_utils, etc.) resolve.
    """
    if _VENDOR_PATH not in sys.path:
        sys.path.insert(0, _VENDOR_PATH)

    cls_name = platform.capitalize()  # Amazon, Shein, Shopee, AliExpress, MercadoLivre
    # Special cases:
    if cls_name == "Mercadolivre":
        cls_name = "MercadoLivre"
    if cls_name == "Aliexpress":
        cls_name = "AliExpress"

    mod_name = cls_name  # file is named Amazon.py, Shein.py, etc.
    import importlib
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def _suppress_global_side_effects() -> None:
    """The ecommerce-scraper repo redirects sys.stdout to a Logger at
    import time and plays notification sounds. Suppress these to avoid
    polluting our agent's terminal output.

    This should be called before any scraper imports happen.
    """
    # The Logger redirect is already done at import time. To avoid it
    # taking over our agent's stdout, we wrap it so writes to Logger
    # are discarded during agent runs.
    import logging
    logging.getLogger().setLevel(logging.ERROR)


def _scrape_sync(url: str, platform: str, output_dir: str) -> Dict[str, Any]:
    """Synchronous wrapper that runs inside asyncio.to_thread().

    Calls the repo's scraper.scrape() with suppressed side effects.
    Returns the raw product_data dict from their scrape() method.
    """
    _suppress_global_side_effects()

    # Set env vars the scrapers expect
    os.environ.setdefault("HEADLESS", "True")
    os.environ.setdefault("CHROME_PROFILE_PATH", str(
        Path.home() / ".mozilla" / "edge_profile"
    ))

    ScraperClass = _import_scraper(platform)
    # They want a prefix for output directory naming (e.g. "Amazon")
    prefix = PLATFORMS.get(platform, platform)
    scraper = ScraperClass(url=url, prefix=prefix, output_directory=output_dir)
    try:
        result = scraper.scrape()
        return result or {}
    finally:
        try:
            scraper.close_browser()
        except Exception:
            pass


def _to_product_payload(
    raw: Dict[str, Any],
    url: str,
    platform: str,
) -> ProductPayload:
    """Map the repo's output dict to our ProductPayload domain entity.

    Their scrape() returns a dict with (at minimum):
        name, current_price (str), current_price_integer,
        current_price_decimal, old_price, old_price_integer,
        old_price_decimal, discount_percentage, description,
        downloaded_files (list of str paths)
    """
    title = raw.get("name", "Unknown Product")

    # Price as a single variant
    price_str = raw.get("current_price", "0")
    price_int = int(raw.get("current_price_integer", 0))
    price_dec = int(raw.get("current_price_decimal", 0))
    price = float(f"{price_int}.{price_dec:02d}")

    variant = ItemVariant(
        sku=f"{platform}-{hash(url)}",
        label=f"Standard - {price_str}",
        price=price,
        in_stock=True,
    )

    # Everything else goes into dynamic_attributes
    dynamic: Dict[str, Any] = {
        "platform": platform,
        "source": "E-Commerces-WebScraper",
        "current_price": raw.get("current_price", ""),
        "old_price": raw.get("old_price", ""),
        "discount_percentage": raw.get("discount_percentage", ""),
        "description": raw.get("description", ""),
    }

    # Clean up None/empty values
    dynamic = {k: v for k, v in dynamic.items() if v}

    # If product images were downloaded, note their paths
    downloaded = raw.get("downloaded_files", [])
    if downloaded:
        dynamic["images"] = downloaded

    return ProductPayload(
        title=title,
        source_url=url,
        variants=[variant],
        reviews=[],
        dynamic_attributes=dynamic,
    )


class ECommerceAdapter:
    """Async wrapper around the sync ecommerce scraper submodule.

    Usage:
        adapter = ECommerceAdapter()
        payload = await adapter.fetch_product("https://amazon.com/dp/...")

    The raw scrape runs in a thread (asyncio.to_thread) so the async
    agent loop stays responsive.
    """

    def __init__(self, output_dir: str | None = None) -> None:
        self._output_dir = output_dir or os.path.join(
            tempfile.gettempdir(), "retail-scraper-outputs"
        )
        os.makedirs(self._output_dir, exist_ok=True)

    async def fetch_product(
        self,
        url: str,
        platform: str | None = None,
    ) -> ProductPayload:
        """Fetch product data from an ecommerce URL.

        Args:
            url: Product page URL (Amazon, AliExpress, etc.)
            platform: Override auto-detection (e.g. "amazon", "aliexpress").
                      If None, we try all known platforms' scrapers.

        Returns:
            ProductPayload with extracted title, price, and metadata.
        """
        loop = asyncio.get_event_loop()

        if platform is not None:
            raw = await loop.run_in_executor(
                None, _scrape_sync, url, platform, self._output_dir
            )
            return _to_product_payload(raw, url, platform)

        # Auto-detect: try each platform's scraper
        # We check the URL for domain hints
        url_lower = url.lower()
        for pname, display in PLATFORMS.items():
            if pname in url_lower:
                try:
                    raw = await loop.run_in_executor(
                        None, _scrape_sync, url, pname, self._output_dir
                    )
                    if raw and raw.get("name"):
                        return _to_product_payload(raw, url, pname)
                except Exception:
                    continue

        raise ValueError(
            f"Could not scrape {url}. "
            f"Supported platforms: {', '.join(PLATFORMS.values())}"
        )

    async def fetch_json(self, url: str, platform: str | None = None) -> str:
        """Convenience: fetch a product and return the payload as JSON."""
        payload = await self.fetch_product(url, platform)
        return json.dumps(
            {
                "title": payload.title,
                "source_url": payload.source_url,
                "price": payload.variants[0].price if payload.variants else 0,
                "price_label": payload.variants[0].label if payload.variants else "",
                "dynamic_attributes": payload.dynamic_attributes,
            },
            ensure_ascii=False,
        )