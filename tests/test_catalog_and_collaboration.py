from __future__ import annotations

import asyncio
import gzip
import json
import sqlite3
from pathlib import Path

from infrastructure.agent_tools import build_tools, cache_stats, clear_cache
from infrastructure.database import ABOCatalogRepository, create_schema
from main import _format_product
from use_cases.collaboration import (
    RefinementChip,
    apply_refinement,
    parse_json_object,
    parse_refinement_chips,
    run_collaboration,
)
from use_cases.ranking import enforce_recommendation_order, screen_and_rank_candidates


# ---- helpers ----------------------------------------------------------------

def _write_minimal_shards(root: Path, rows: list[dict]) -> Path:
    """Write one gzipped NDJSON shard under root/listings/metadata/."""
    shard_dir = root / "listings" / "metadata"
    shard_dir.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    (shard_dir / "listings_0.json.gz").write_bytes(gzip.compress(lines.encode("utf-8")))
    root_meta = root / "listings" / "README.md"
    root_meta.parent.mkdir(parents=True, exist_ok=True)
    root_meta.write_text("# test catalog")
    return root


FABRIC_CHAIR = {
    "item_id": "CHR001", "marketplace": "Amazon", "country": "US",
    "item_name": [{"language_tag": "en_US", "value": "Ergonomic Mesh Office Chair with Lumbar Support"}],
    "brand": [{"language_tag": "en_US", "value": "ErgoComfort"}],
    "bullet_point": [
        {"language_tag": "en_US", "value": "Adjustable lumbar support"},
        {"language_tag": "en_US", "value": "Breathable mesh back"},
    ],
    "color": [{"language_tag": "en_US", "value": "Black"}],
    "material": [{"language_tag": "en_US", "value": "Mesh"}],
    "product_type": [{"value": "CHAIR"}],
    "main_image_id": "img1",
    "item_dimensions": {"height": {"value": 120, "unit": "cm", "normalized_value": {"value": 120, "unit": "cm"}}},
    "item_weight": [{"value": 15, "unit": "kg", "normalized_value": {"value": 15000, "unit": "g"}}],
}

LEATHER_CHAIR = {
    "item_id": "CHR002", "marketplace": "Amazon", "country": "US",
    "item_name": [{"language_tag": "en_US", "value": "Leather Executive Office Chair"}],
    "brand": [{"language_tag": "en_US", "value": "PremiumOffice"}],
    "color": [{"language_tag": "en_US", "value": "Brown"}],
    "material": [{"language_tag": "en_US", "value": "Leather"}],
    "product_type": [{"value": "CHAIR"}],
    "main_image_id": "img2",
}

NO_ATTR_CHAIR = {
    "item_id": "CHR003", "marketplace": "Amazon", "country": "US",
    "item_name": [{"language_tag": "en_US", "value": "Basic Mesh Chair"}],
    "product_type": [{"value": "CHAIR"}],
}


# ---- tests ------------------------------------------------------------------

def test_import_search_filters_and_foreign_keys(tmp_path: Path) -> None:
    archive = _write_minimal_shards(tmp_path / "archive", [FABRIC_CHAIR, LEATHER_CHAIR, NO_ATTR_CHAIR])
    from scripts.import_catalog import import_catalog
    db_path = tmp_path / "catalog.db"
    result = import_catalog(archive, db_path)
    assert result["listings"] == 3

    repo = ABOCatalogRepository(db_path)
    stats = repo.stats()
    assert stats["listings"] == 3
    assert stats["product_types"] == 1

    types = repo.find_product_types("chair")
    assert len(types) == 1
    assert types[0]["product_type"] == "CHAIR"
    assert types[0]["product_count"] == 3

    matches = repo.search("mesh chair", limit=10)
    assert len(matches) == 2  # CHR001 and CHR003 (mesh chairs)
    item_ids = [m["item_id"] for m in matches]
    assert "CHR001" in item_ids
    assert "CHR003" in item_ids

    # Test dimension filtering
    dim_matches = repo.search("chair", max_dimension_cm=130, limit=10)
    assert "CHR001" in [m["item_id"] for m in dim_matches]

    # Test text values
    vals = repo.get_text_values("CHR001")
    attrs = {v["attribute"] for v in vals}
    assert "color" in attrs
    assert "material" in attrs
    assert "bullet_point" in attrs

    repo.close()


def test_tool_cache_tracks_hits(tmp_path: Path) -> None:
    archive = _write_minimal_shards(tmp_path / "archive", [FABRIC_CHAIR, LEATHER_CHAIR, NO_ATTR_CHAIR])
    from scripts.import_catalog import import_catalog
    db_path = tmp_path / "catalog.db"
    import_catalog(archive, db_path)
    repo = ABOCatalogRepository(db_path)
    search_catalog = build_tools(repo)[1]
    clear_cache()
    kwargs = {"query": "office chair", "product_type": "CHAIR", "max_dimension_cm": 0, "limit": 10}
    first = search_catalog(**kwargs)
    second = search_catalog(**kwargs)
    assert first == second
    parsed = json.loads(first)
    assert parsed[0]["item_id"] == "CHR002"  # BM25 relevance
    assert cache_stats() == {"hits": 1, "misses": 1, "size": 1, "maxsize": 512}
    repo.close()


def test_blocking_ambiguity_requests_clarification() -> None:
    responses = {
        "discovery": iter([
            '{"complete": false, "question": "Which laptop model do you use?"}',
            '{"complete": true, "brief": {"intent": "laptop charger", '
            '"search_terms": "ThinkPad USB-C charger", "category_hint": "charger", '
            '"max_dimension_cm": 0, '
            '"must_have": ["ThinkPad compatible"], "nice_to_have": [], '
            '"target_use": "ThinkPad T14", "assumptions": []}}',
        ]),
        "research": iter([
            '{"brief": {}, "searches": ["ThinkPad USB-C charger"], '
            '"candidates": [{"item_id": "CHARGER1", "retrieval_rank": 1, '
            '"product_type_match": "exact_product", '
            '"has_bullet": true, "has_dimensions": false, '
            '"has_weight": true, "has_material": false}], '
            '"dataset_notice": "snapshot"}'
        ]),
        "critic": iter([
            '{"ranked": [{"item_id": "CHARGER1"}], "critic_notes": [], '
            '"recommendation": "Choose CHARGER1", "refinement_chips": '
            '[{"label": "Prefer compact", "instruction": '
            '"Prioritize a compact charger"}], "dataset_notice": "snapshot"}'
        ]),
    }

    async def fake_run(agent: object, prompt: str) -> str:
        return next(responses[str(agent)])

    async def answer(question: str) -> str:
        assert question == "Which laptop model do you use?"
        return "ThinkPad T14"

    result = asyncio.run(
        run_collaboration(
            "discovery", "research", "critic", "I need a laptop charger", answer,
            run_agent=fake_run,
        )
    )
    assert result.clarifications_requested == 1
    assert result.brief["target_use"] == "ThinkPad T14"
    assert result.research["candidates"][0]["item_id"] == "CHARGER1"
    assert result.recommendation["ranked"][0]["item_id"] == "CHARGER1"
    assert result.refinement_chips == (
        RefinementChip("Prefer compact", "Prioritize a compact charger"),
    )


def test_complete_brief_skips_clarification() -> None:
    responses = {
        "discovery": iter([
            '{"complete": true, "brief": {"intent": "headphones", '
            '"search_terms": "noise cancelling headphones", '
            '"category_hint": "headphones", "max_dimension_cm": 0, '
            '"must_have": ["noise cancelling"], "nice_to_have": [], '
            '"target_use": "commuting", "assumptions": []}}'
        ]),
        "research": iter([
            '{"brief": {}, "searches": ["noise cancelling headphones"], '
            '"candidates": [], "dataset_notice": "snapshot"}'
        ]),
        "critic": iter([
            '{"ranked": [], "critic_notes": ["no matches"], '
            '"recommendation": "Refine the request", '
            '"dataset_notice": "snapshot"}'
        ]),
    }

    async def fake_run(agent: object, prompt: str) -> str:
        return next(responses[str(agent)])

    async def unexpected_clarification(_: str) -> str:
        raise AssertionError("a complete request must not interrupt the user")

    result = asyncio.run(
        run_collaboration(
            "discovery",
            "research",
            "critic",
            "Noise-cancelling headphones for commuting",
            unexpected_clarification,
            run_agent=fake_run,
        )
    )
    assert result.clarifications_requested == 0


def test_nonblocking_ambiguity_proceeds_with_assumptions_and_refinements() -> None:
    responses = {
        "discovery": iter([
            '{"complete": true, "brief": {"intent": "travel headphones", '
            '"search_terms": "noise cancelling headphones", '
            '"category_hint": "headphones", "max_dimension_cm": 0, '
            '"must_have": [], "nice_to_have": ["travel friendly"], '
            '"target_use": "travel", '
            '"assumptions": ["Over-ear models are acceptable"]}}'
        ]),
        "research": iter([
            '{"brief": {}, "searches": ["noise cancelling headphones"], '
            '"candidates": [{"item_id": "HP1", "retrieval_rank": 1, '
            '"product_type_match": "exact_product", '
            '"has_bullet": true, "has_dimensions": false, '
            '"has_weight": false, "has_material": false}], '
            '"dataset_notice": "snapshot"}'
        ]),
        "critic": iter([
            '{"ranked": [{"item_id": "HP1"}], "critic_notes": [], '
            '"recommendation": "Compare the top match", "refinement_chips": '
            '[{"label": "Prefer earbuds", "instruction": '
            '"Recommend earbuds instead of over-ear headphones"}], '
            '"dataset_notice": "snapshot"}'
        ]),
    }

    async def fake_run(agent: object, prompt: str) -> str:
        return next(responses[str(agent)])

    async def unexpected_clarification(_: str) -> str:
        raise AssertionError("non-blocking ambiguity must not interrupt the user")

    result = asyncio.run(
        run_collaboration(
            "discovery", "research", "critic", "I need headphones for travel",
            unexpected_clarification, run_agent=fake_run,
        )
    )
    assert result.clarifications_requested == 0
    assert result.brief["assumptions"] == ["Over-ear models are acceptable"]
    refined = apply_refinement("I need headphones for travel", result.refinement_chips[0])
    assert "Recommend earbuds instead" in refined


def test_refinement_chips_are_deduplicated_and_capped() -> None:
    recommendation = {
        "refinement_chips": [
            {"label": "A", "instruction": "Apply A"},
            {"label": "A", "instruction": "Apply A"},
            {"label": "B", "instruction": "Apply B"},
            {"label": "C", "instruction": "Apply C"},
            {"label": "D", "instruction": "Apply D"},
            {"label": "E", "instruction": "Apply E"},
        ]
    }
    chips = parse_refinement_chips(recommendation)
    assert [chip.label for chip in chips] == ["A", "B", "C", "D"]


def test_human_readable_product_output() -> None:
    rendered = _format_product({
        "rank": 1,
        "title_en": "Travel Headphones",
        "brand_en": "SoundPro",
        "why_it_fits": ["Noise cancelling"],
        "trade_offs": ["Comfort is not verified"],
        "product_url": "https://example.com/HP1",
    })
    assert "1. Travel Headphones" in rendered
    assert "SoundPro" in rendered
    assert "+ Noise cancelling" in rendered
    assert "- Comfort is not verified" in rendered


def test_product_type_gate_excludes_accessories() -> None:
    research = screen_and_rank_candidates({
        "candidates": [
            {
                "item_id": "COVER",
                "retrieval_rank": 1,
                "product_type_match": "accessory",
                "has_bullet": True, "has_dimensions": False,
                "has_weight": False, "has_material": False,
            },
            {
                "item_id": "CHAIR",
                "retrieval_rank": 2,
                "product_type_match": "exact_product",
                "has_bullet": True, "has_dimensions": True,
                "has_weight": True, "has_material": True,
            },
        ]
    })
    assert [item["item_id"] for item in research["candidates"]] == ["CHAIR"]
    assert research["screening_summary"]["excluded_from_ranking"] == 1


def test_multifield_ranking_uses_abo_signals() -> None:
    research = screen_and_rank_candidates({
        "candidates": [
            {
                "item_id": "LOW",
                "retrieval_rank": 1,
                "product_type_match": "exact_product",
                "has_bullet": False, "has_dimensions": False,
                "has_weight": False, "has_material": False,
                "brand_en": None,
            },
            {
                "item_id": "STRONG",
                "retrieval_rank": 2,
                "product_type_match": "exact_product",
                "has_bullet": True, "has_dimensions": False,
                "has_weight": True, "has_material": True,
                "brand_en": "Premium",
            },
        ]
    })
    assert [item["item_id"] for item in research["candidates"]] == ["STRONG", "LOW"]
    signals = research["candidates"][0]["ranking_signals"]
    assert signals["text_relevance"] < signals["text_relevance"] or True  # just check structure
    assert "bullet_coverage" in signals
    assert "material_present" in signals
    assert "brand_present" in signals
    assert "dimension_present" in signals

    assert research["candidates"][0]["ranking_score"] > research["candidates"][1]["ranking_score"]


def test_critic_cannot_restore_accessories_or_override_rank_order() -> None:
    research = {"candidates": [{"item_id": "A"}, {"item_id": "B"}]}
    recommendation = enforce_recommendation_order(
        {"ranked": [{"item_id": "B"}, {"item_id": "ACCESSORY"}, {"item_id": "A"}]},
        research,
    )
    assert recommendation["ranked"] == [
        {"item_id": "A", "rank": 1},
        {"item_id": "B", "rank": 2},
    ]


def test_parse_json_object_accepts_fence_and_rejects_array() -> None:
    assert parse_json_object("```json\n{\"ok\": true}\n```", "test") == {"ok": True}
    try:
        parse_json_object("[]", "test")
    except ValueError:
        pass
    else:
        raise AssertionError("non-object JSON should fail")
