"""Domain layer: pure entities and contracts. No framework imports."""
from .entities import CatalogProduct
from .recommendation import (
    FinalizedCandidate,
    MAX_RANKED_PRODUCTS,
    MAX_REFINEMENT_CHIPS,
    RankedItem,
    RecommendationResponse,
    RefinementChip,
)

__all__ = [
    "CatalogProduct",
    "FinalizedCandidate",
    "MAX_RANKED_PRODUCTS",
    "MAX_REFINEMENT_CHIPS",
    "RankedItem",
    "RecommendationResponse",
    "RefinementChip",
]