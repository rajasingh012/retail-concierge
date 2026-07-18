"""In-memory BM25 exact-keyword index over the SQLite catalog.

BM25Okapi gives us high-precision exact-token matches — important
for brand names like 'Steelcase' or 'Herman Miller' that vector
retrieval tends to dilute.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

from rank_bm25 import BM25Okapi

from infrastructure.database import ProductCatalogRepository

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercased alphanumeric tokens — keeps brand-name precision high."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class Hit:
    product_id: str
    title: str
    score: float


class LocalHybridSearchEngine:
    """Loads the catalog into a BM25Okapi corpus once, then searches in-memory."""

    def __init__(self, repo: ProductCatalogRepository) -> None:
        self._repo = repo
        self._corpus: List[List[str]] = []
        self._ids: List[str] = []
        self._titles: List[str] = []
        self._bm25: BM25Okapi | None = None
        self.reload()

    # ---------- corpus lifecycle ----------

    def reload(self) -> None:
        products = self._repo.all()
        self._ids = [p.source_url for p in products]  # product_id == source_url
        self._titles = [p.title for p in products]
        self._corpus = [_tokenize(p.full_text()) for p in products]
        # Empty corpus is legal; search() will short-circuit on it.
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None

    # ---------- queries ----------

    def search(self, query: str, limit: int = 10) -> List[Hit]:
        """Return top-`limit` products ranked by BM25 token overlap.

        Empty queries return an empty list. Tokens with zero document
        frequency still get a score (BM25Okapi's standard behavior)
        but contribute nothing useful, which is fine — we want exact
        brand names to dominate.
        """
        if not self._bm25 or not query.strip():
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda pair: pair[1], reverse=True
        )[:limit]
        return [
            Hit(product_id=self._ids[i], title=self._titles[i], score=float(s))
            for i, s in ranked
            if s > 0
        ]

    # ---------- diagnostics ----------

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)