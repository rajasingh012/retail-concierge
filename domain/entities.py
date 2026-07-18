"""Pure domain entities for the RetailConcierge system.

No framework or infrastructure imports — these are the canonical
data contracts that the use_cases and infrastructure layers
must respect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class ItemVariant:
    """A specific purchasable variant of a product (size, color, sku, etc.)."""

    sku: str
    label: str
    price: float
    in_stock: bool = True
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductPayload:
    """Aggregate product record assembled from a marketplace source.

    `dynamic_attributes` is the catch-all bucket for arbitrary
    marketplace fields we don't model explicitly (badges, ratings,
    shipping windows, etc.).
    """

    title: str
    source_url: str
    variants: List[ItemVariant] = field(default_factory=list)
    reviews: List[str] = field(default_factory=list)
    dynamic_attributes: Dict[str, Any] = field(default_factory=dict)

    def full_text(self) -> str:
        """Flat text used by the keyword indexer."""
        parts: List[str] = [self.title]
        brand = self.dynamic_attributes.get("brand")
        if brand:
            parts.append(str(brand))
        description = self.dynamic_attributes.get("description")
        if description:
            parts.append(str(description))
        for v in self.variants:
            parts.append(v.label)
        return " ".join(parts)