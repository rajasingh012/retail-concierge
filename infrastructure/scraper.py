"""Playwright scraper with persistent browser context.

Targets a user data dir so cookies/proxy state survive across
runs and bot-telemetry defenses see a more stable fingerprint.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from playwright.async_api import BrowserContext, async_playwright

from domain.entities import ItemVariant, ProductPayload


DEFAULT_USER_DATA_DIR = "/home/rajasingh/.mozilla/edge_profile"


class PlaywrightScraper:
    """Async Playwright wrapper. Holds one persistent chromium context."""

    def __init__(
        self,
        user_data_dir: str | Path = DEFAULT_USER_DATA_DIR,
        headless: bool = True,
        locale: str = "en-US",
    ) -> None:
        self._user_data_dir = str(user_data_dir)
        self._headless = headless
        self._locale = locale
        self._pw = None
        self._context: Optional[BrowserContext] = None

    async def __aenter__(self) -> "PlaywrightScraper":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def start(self) -> None:
        Path(self._user_data_dir).mkdir(parents=True, exist_ok=True)
        self._pw = await async_playwright().start()
        self._context = await self._pw.chromium.launch_persistent_context(
            user_data_dir=self._user_data_dir,
            headless=self._headless,
            locale=self._locale,
            # Bot-telemetry mitigation: realistic UA + viewport. Stealth plugins
            # are out of scope here — caller can inject init scripts later.
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1366, "height": 900},
        )

    async def close(self) -> None:
        if self._context is not None:
            await self._context.close()
        if self._pw is not None:
            await self._pw.stop()
        self._context = None
        self._pw = None

    # ---------- public API ----------

    async def fetch_html(self, url: str, wait_selector: str | None = None) -> str:
        if self._context is None:
            raise RuntimeError("Scraper not started — call start() or use 'async with'.")
        page = await self._context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            if wait_selector:
                await page.wait_for_selector(wait_selector, timeout=15_000)
            return await page.content()
        finally:
            await page.close()

    async def fetch_product(
        self, url: str, selectors: Dict[str, str]
    ) -> ProductPayload:
        """Generic product fetch using caller-supplied CSS selectors.

        Recognized keys:
            title, brand, description, review
        Variant rows use `variant_row` plus `variant_sku`, `variant_label`,
        `variant_price`, `variant_in_stock`.
        """
        if self._context is None:
            raise RuntimeError("Scraper not started — call start() or use 'async with'.")
        page = await self._context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45_000)

            title = (await page.locator(selectors["title"]).first.inner_text()).strip()
            brand = ""
            if selectors.get("brand"):
                try:
                    brand = (await page.locator(selectors["brand"]).first.inner_text()).strip()
                except Exception:
                    pass
            description = ""
            if selectors.get("description"):
                try:
                    description = (await page.locator(selectors["description"]).first.inner_text()).strip()
                except Exception:
                    pass
            reviews: list[str] = []
            if selectors.get("review"):
                reviews = await page.locator(selectors["review"]).all_inner_texts()

            variants: list[ItemVariant] = []
            if selectors.get("variant_row"):
                rows = page.locator(selectors["variant_row"])
                count = await rows.count()
                for i in range(count):
                    row = rows.nth(i)
                    sku = (await row.locator(selectors["variant_sku"]).inner_text()).strip()
                    label = (await row.locator(selectors["variant_label"]).inner_text()).strip()
                    price_text = (await row.locator(selectors["variant_price"]).inner_text()).strip()
                    in_stock = True
                    if selectors.get("variant_in_stock"):
                        try:
                            in_stock = (
                                await row.locator(selectors["variant_in_stock"]).is_visible()
                            )
                        except Exception:
                            pass
                    price = _parse_price(price_text)
                    variants.append(
                        ItemVariant(
                            sku=sku,
                            label=label,
                            price=price,
                            in_stock=in_stock,
                        )
                    )

            dynamic: Dict[str, Any] = {}
            if brand:
                dynamic["brand"] = brand
            if description:
                dynamic["description"] = description

            return ProductPayload(
                title=title,
                source_url=url,
                variants=variants,
                reviews=[r.strip() for r in reviews if r.strip()],
                dynamic_attributes=dynamic,
            )
        finally:
            await page.close()


def _parse_price(text: str) -> float:
    """Best-effort price parser: '$1,299.00' -> 1299.00."""
    digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    return float(digits) if digits else 0.0


# Convenience: dump a fetched product as JSON for piping into other tools.
def payload_to_json(payload: ProductPayload) -> str:
    return json.dumps(
        {
            "title": payload.title,
            "source_url": payload.source_url,
            "variants": [v.__dict__ for v in payload.variants],
            "reviews": payload.reviews,
            "dynamic_attributes": payload.dynamic_attributes,
        },
        ensure_ascii=False,
        indent=2,
    )