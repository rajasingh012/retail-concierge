"""MAF ContextProvider that injects the catalog vocabulary per turn.

This module follows Microsoft's documented "Context Engineering"
pattern for Microsoft Agent Framework (MAF):

    - ADR  docs/decisions/0016-python-context-middleware.md
      Defines ``ContextProvider`` as the canonical abstraction for
      "Injects instructions, messages, and tools before/after
      invocations" and renames the legacy ``AIContextProvider.InvokingAsync``
      hook from .NET into Python's ``ContextProvider.before_run`` that
      mutates a per-invocation ``SessionContext``.

    - Sample  python/samples/02-agents/context_providers/
                  simple_context_provider.py
      Canonical ``UserInfoMemory`` example: keeps the static
      ``instructions=`` tiny and uses ``extend_instructions`` for
      per-session, state-dependent content (the user's name/age).

We use the same shape: the static ``instructions=`` stays a small
house-rules string, and the canonical ``product_type`` list reaches
the model via ``SessionContext.extend_instructions`` on every
``before_run`` call. Brands are NOT injected — the ABO catalog is
dominated by Amazon private-label names that real shoppers do not
search by, and DeepSeek already knows common brand names. Brand
canonicalization happens at search time via ``find_brands``
(FTS5 + LIKE fallback).

Pydantic still enforces the same product_type vocabulary on the way
in (``domain.recommendation.set_catalog_vocabulary`` /
``ShoppingBrief._gate_against_catalog_vocabulary``). The prompt
injection is a HINT to the LLM, the validator is the GATE. They
share the same catalog vocabulary list at process-boot time.
"""
from __future__ import annotations

from agent_framework import ContextProvider

# Source attribution string used in both the SessionContext mutation
# and any log lines. Stable across releases so debugging output stays
# greppable.
CATALOG_VOCAB_SOURCE_ID = "catalog_vocabulary"


class CatalogVocabularyProvider(ContextProvider):
    """Per-invocation injection of canonical ``product_type``.

    Subclasses MAF's ``ContextProvider`` and overrides ``before_run``.
    On every model call it formats the configured catalog vocabulary
    and emits it via ``context.extend_instructions(source_id, body)``.

    Notes:
        - Empty vocabulary -> no-op (does not pollute the prompt with
          empty section labels).
        - The provider holds the vocabulary in memory (no DB lookup
          per call). It is built once per ``Agent`` instance, which
          aligns with the upstream static-vocabulary behavior.
        - No per-session caching here. That is a future optimization
          to consider only after measuring OpenAI/Anthropic
          ``prompt_cache_key`` benefit on the target models.
    """

    def __init__(
        self,
        *,
        product_types: list[str],
    ) -> None:
        super().__init__(source_id=CATALOG_VOCAB_SOURCE_ID)
        # Filter empty / whitespace-only entries so a half-loaded
        # catalog (e.g. importer run that hasn't populated types yet)
        # produces a clean provider that doesn't churn empty strings
        # into the prompt.
        self._product_types = [
            str(t) for t in product_types if t and str(t).strip()
        ]

    @property
    def product_types(self) -> list[str]:
        return list(self._product_types)

    def format_body(self) -> str:
        """Render the vocabulary as the model will see it.

        Exposed publicly so tests can assert exact text without
        faking the SessionContext.
        """
        types_section = _format_section(
            "CATALOG_PRODUCT_TYPES", self._product_types
        )
        return f"Catalog vocabulary:\n{types_section}"

    async def before_run(
        self,
        *,
        agent,
        session,
        context,
        state,
    ) -> None:
        """MAF hook. Called before every model invocation.

        Skips injection when there is nothing to say (empty catalog
        at process boot is a legitimate, harmless state).
        """
        body = self.format_body()
        if not body_has_content(body):
            return
        context.extend_instructions(CATALOG_VOCAB_SOURCE_ID, body)


def _format_section(label: str, terms: list[str]) -> str:
    """Format a single vocabulary section. Empty -> short placeholder.

    Matching the previous static-bake style so model behavior is
    unchanged when the provider is in place.
    """
    if not terms:
        return f"{label}: (catalog has no entries yet)"
    body = ", ".join(terms)
    return f"{label}:\n  {body}"


def body_has_content(body: str) -> bool:
    """True iff the formatted body carries any real vocabulary line.

    Both placeholder lines ('(catalog has no entries yet)') are
    treated as empty for the purpose of injection: there is no
    information for the model to gain from seeing placeholders.
    """
    if not body:
        return False
    placeholders = (
        "CATALOG_PRODUCT_TYPES: (catalog has no entries yet)",
    )
    return not all(p in body for p in placeholders)
