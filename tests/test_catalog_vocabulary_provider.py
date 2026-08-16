"""Tests for the MAF ContextProvider that injects catalog vocabulary.

These tests pin down the MAF-canonical pattern documented in
``docs/decisions/0016-python-context-middleware.md`` and demonstrated by
``python/samples/02-agents/context_providers/simple_context_provider.py``
in the upstream microsoft/agent-framework repo.

We assert three contracts:

  1. ``before_run`` calls ``SessionContext.extend_instructions`` with a
     stable ``source_id`` and the formatted vocabulary body when at
     least one vocabulary section has real content.

  2. When the vocabulary is empty (catalog has no entries yet),
     ``before_run`` is a no-op — it does NOT inject a placeholder
     section, which would otherwise pollute the prompt.

  3. ``build_shopping_agent`` no longer bakes the catalog labels
     (``CATALOG_PRODUCT_TYPES:`` / ``CATALOG_BRANDS:``) into the static
     ``instructions=`` payload. Those labels now arrive through the
     provider pipeline at run time, not via the constructor.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from infrastructure.catalog_vocabulary_provider import (
    CATALOG_VOCAB_SOURCE_ID,
    CatalogVocabularyProvider,
    body_has_content,
)


# ---------------------------------------------------------------------------
# Provider-level tests
# ---------------------------------------------------------------------------


class _RecordingContext:
    """Stand-in for ``SessionContext`` that records ``extend_instructions`` calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def extend_instructions(self, source_id: str, body) -> None:
        # MAF accepts str | Sequence[str]; we always pass a single str.
        self.calls.append((source_id, str(body)))


def _run_before_run(provider: CatalogVocabularyProvider) -> _RecordingContext:
    """Invoke the provider's MAF hook with a recording context."""
    import asyncio

    ctx = _RecordingContext()
    asyncio.run(
        provider.before_run(
            agent=MagicMock(name="agent"),
            session=MagicMock(name="session"),
            context=ctx,
            state={},
        )
    )
    return ctx


def test_source_id_is_stable_source_attribution():
    """The source_id doubles as a stable attribution key for debugging.

    Changing it would invalidate log greps and would not match the
    upstream ``samples/.../simple_context_provider.py`` convention.
    """
    assert CATALOG_VOCAB_SOURCE_ID == "catalog_vocabulary"
    provider = CatalogVocabularyProvider(product_types=[], brands=[])
    assert provider.source_id == CATALOG_VOCAB_SOURCE_ID


def test_before_run_injects_vocabulary_via_extend_instructions():
    """Happy path: vocabulary reaches the model via the MAF pipeline."""
    provider = CatalogVocabularyProvider(
        product_types=["chair", "sofa"],
        brands=["IKEA", "Herman Miller"],
    )

    ctx = _run_before_run(provider)

    assert len(ctx.calls) == 1, "before_run should emit exactly one extension"
    source_id, body = ctx.calls[0]
    assert source_id == CATALOG_VOCAB_SOURCE_ID
    # The body must contain both section labels and the terms.
    assert "CATALOG_PRODUCT_TYPES:" in body
    assert "chair" in body
    assert "sofa" in body
    assert "CATALOG_BRANDS:" in body
    assert "IKEA" in body
    assert "Herman Miller" in body


def test_before_run_skips_injection_when_vocabulary_is_empty():
    """No entries -> don't append empty placeholder sections."""
    provider = CatalogVocabularyProvider(product_types=[], brands=[])

    ctx = _run_before_run(provider)

    assert ctx.calls == [], (
        "before_run must be a no-op when neither product_types nor "
        "brands have any real entries"
    )


def test_format_body_uses_placeholder_when_section_is_empty():
    """Format still produces placeholders for an empty section, but
    ``body_has_content`` flags the whole body as no-op so the provider
    skips emission. Two layers of defense: predictable format for
    debugging, predictable skip for the model."""
    provider = CatalogVocabularyProvider(product_types=[], brands=[])

    body = provider.format_body()
    assert "CATALOG_PRODUCT_TYPES: (catalog has no entries yet)" in body
    assert "CATALOG_BRANDS: (catalog has no entries yet)" in body
    assert body_has_content(body) is False


def test_format_body_has_content_when_any_section_has_entries():
    """One real section is enough — the model gets the hint."""
    provider = CatalogVocabularyProvider(
        product_types=["chair"], brands=[]
    )
    body = provider.format_body()
    assert body_has_content(body) is True


def test_provider_filters_blank_entries():
    """Blank strings and empty list entries are dropped at construction time."""
    provider = CatalogVocabularyProvider(
        product_types=["chair", "", "  ", "sofa"],
        brands=["IKEA", "", "  ", "Herman Miller"],
    )
    assert provider.product_types == ["chair", "sofa"]
    assert provider.brands == ["IKEA", "Herman Miller"]


# ---------------------------------------------------------------------------
# Agent-wiring tests (catalog labels must NOT leak into static instructions)
# ---------------------------------------------------------------------------


def test_build_shopping_agent_no_longer_bakes_catalog_labels_into_instructions():
    """The 80/2000 static-bake path is gone. Catalog labels now flow via
    ``context_providers`` only. This is a regression guard so the
    static bake doesn't sneak back in."""
    from unittest.mock import MagicMock

    from use_cases.shopping_agent import build_shopping_agent

    catalog_vocab = {
        "product_types": ["chair", "sofa", "desk"],
        "brands": ["IKEA", "Herman Miller", "Steelcase"],
    }
    agent = build_shopping_agent(
        client=MagicMock(name="client"),
        catalog_tools=[],
        catalog_vocabulary=catalog_vocab,
    )
    # In MAF v1.13 the constructor-baked instructions land on
    # ``agent.default_options["instructions"]`` (verified via
    # ``Agent.__init__`` source inspection). The provider-injected
    # section lives in the MAF context-provider pipeline, not here.
    instructions = agent.default_options.get("instructions", "") or ""

    assert "CATALOG_PRODUCT_TYPES:" not in instructions, (
        "Static instructions must not contain catalog labels — the "
        "ContextProvider is responsible for that injection now."
    )
    assert "CATALOG_BRANDS:" not in instructions, (
        "Same regression guard for the brands section."
    )


def test_build_shopping_agent_registers_a_context_provider():
    """A ``CatalogVocabularyProvider`` is the entry point for the
    canonical MAF pattern (``Agent(context_providers=[...])``)."""
    from unittest.mock import MagicMock

    from use_cases.shopping_agent import build_shopping_agent

    agent = build_shopping_agent(
        client=MagicMock(name="client"),
        catalog_tools=[],
        catalog_vocabulary={
            "product_types": ["chair"],
            "brands": ["IKEA"],
        },
    )

    # MAF exposes providers via a private attr in 1.13; check
    # both the documented Agent-level knob and the underlying list.
    providers = list(getattr(agent, "context_providers", []) or [])
    assert any(
        isinstance(p, CatalogVocabularyProvider) for p in providers
    ), f"expected CatalogVocabularyProvider among context_providers, got {providers!r}"


def test_build_shopping_agent_with_empty_vocabulary_registers_noop_provider():
    """An agent built with no vocabulary still gets a registered
    provider (so the context_providers list is shape-stable across
    catalog states). The provider itself no-ops via ``before_run``."""
    from unittest.mock import MagicMock

    from use_cases.shopping_agent import build_shopping_agent

    agent = build_shopping_agent(
        client=MagicMock(name="client"),
        catalog_tools=[],
        catalog_vocabulary=None,
    )
    providers = list(getattr(agent, "context_providers", []) or [])
    assert any(
        isinstance(p, CatalogVocabularyProvider) for p in providers
    )
