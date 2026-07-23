"""Domain contracts for ABO catalog search results."""

from pydantic import BaseModel, Field


class CatalogProduct(BaseModel):
    """Public view of one ABO catalog listing used at the application boundary."""

    item_id: str
    title_en: str = ""
    brand_en: str = ""
    product_type: str = ""
    product_url: str = ""
    marketplace: str = ""
    country: str = ""
    has_bullet: int = 0
    has_dimensions: int = 0
    has_weight: int = 0
    has_material: int = 0