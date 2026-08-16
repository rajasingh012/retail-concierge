"""Project-wide pytest fixtures.

``set_catalog_vocabulary`` mutates module-level state inside
``domain.recommendation`` so that the brief-time Pydantic validator
can flag off-vocabulary product_type values. When a test exercises
the production code path that calls ``_seed_brief_validator`` (for
example via ``build_shopping_agent``), vocabulary persists into the
next test that does not explicitly reset it. That bleeds into
unrelated tests like ``test_extract_brief_tool.py`` which expect
the validator to be unseeded (permissive) so canned fixtures with
arbitrary product_type strings pass.

Brands are NOT in the vocabulary anymore — they are resolved at
search time via ``find_brands``.

The autouse fixture below resets the global vocabulary state before
every test, mirroring the manual reset pattern in
``tests/test_brief_vocabulary_gate.py``. This costs ~microseconds per
test and removes a known class of flakes — both for the existing
suite and for future tests that touch the production seed path.
"""
from __future__ import annotations

import pytest

from domain.recommendation import set_catalog_vocabulary


@pytest.fixture(autouse=True)
def _reset_catalog_vocabulary() -> None:
    """Each test starts with an empty catalog vocabulary.

    Tests that need a seeded vocabulary (e.g. ``test_brief_vocabulary_gate``)
    keep their own per-test seeding; this fixture only guarantees a
    clean baseline so tests that don't care about the validator see
    its permissive-empty state.

    We call ``set_catalog_vocabulary(set())`` once before yielding
    rather than using a yield/cleanup pair, because the goal is to
    start each test from the SAME known-good baseline rather than to
    restore whatever the test produced. Tests that intentionally seed
    (e.g. via ``build_shopping_agent`` then ``model_validate``) end
    with their own seeded state regardless — the next test re-clears,
    which is what we want.
    """
    set_catalog_vocabulary(set())
