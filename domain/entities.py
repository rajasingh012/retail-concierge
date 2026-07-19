"""Pure domain contracts for catalog search results."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CatalogProduct:
    """Evidence returned from the imported Amazon product catalog."""

    id: int
    asin: str
    title: str
    image_url: str
    product_url: str
    stars: float
    review_count: int
    price: float
    list_price: float
    category_id: int
    category_name: str
    is_best_seller: bool
    bought_last_month: int

    def to_dict(self) -> dict[str, Any]:
        """Return the product as a JSON-serializable mapping."""
        return asdict(self)
