from __future__ import annotations

import asyncio
import gzip
import json
from pathlib import Path

from agent_framework import AgentResponse, Content, Message

from domain.recommendation import (
    FinalizedCandidate,
    RecommendationResponse,
    extract_json_object,
)
from infrastructure.agent_tools import build_tools, cache_stats, clear_cache, clamp_limit
from infrastructure.database import ABOCatalogRepository
from main import _format_product
from use_cases.ranking import screen_and_rank_candidates
from use_cases.shopping_agent import (
    EXTRACT_BRIEF_TOOL,
    FINALIZE_RECOMMENDATIONS_TOOL,
    CatalogEvidenceTracker,
    _build_agent_tools,
    _make_finalize_tool,
    enforce_finalized_recommendation,
    finalized_candidates_from_response,
    structured_recommendation_from_response,
)


def _write_minimal_shards(root: Path, rows: list[dict]) -> Path:
    """Write one gzipped NDJSON shard under root/listings/metadata/."""
    shard_dir = root / "listings" / "metadata"
    shard_dir.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    (shard_dir / "listings_0.json.gz").write_bytes(gzip.compress(lines.encode("utf-8")))
    return root


FABRIC_CHAIR = {
    "item_id": "CHR001",
    "marketplace": "Amazon",
    "country": "US",
    "item_name": [
        {
            "language_tag": "en_US",
            "value": "Ergonomic Mesh Office Chair with Lumbar Support",
        }
    ],
    "brand": [{"language_tag": "en_US", "value": "ErgoComfort"}],
    "bullet_point": [
        {"language_tag": "en_US", "value": "Adjustable lumbar support"},
        {"language_tag": "en_US", "value": "Breathable mesh back"},
    ],
    "color": [{"language_tag": "en_US", "value": "Black"}],
    "material": [{"language_tag": "en_US", "value": "Mesh"}],
    "product_type": [{"value": "CHAIR"}],
    "main_image_id": "img1",
    "item_dimensions": {
        "height": {
            "value": 120,
            "unit": "cm",
            "normalized_value": {"value": 120, "unit": "cm"},
        }
    },
    "item_weight": [
        {
            "value": 15,
            "unit": "kg",
            "normalized_value": {"value": 15000, "unit": "g"},
        }
    ],
}

LEATHER_CHAIR = {
    "item_id": "CHR002",
    "marketplace": "Amazon",
    "country": "US",
    "item_name": [
        {"language_tag": "en_US", "value": "Leather Executive Office Chair"}
    ],
    "brand": [{"language_tag": "en_US", "value": "PremiumOffice"}],
    "color": [{"language_tag": "en_US", "value": "Brown"}],
    "material": [{"language_tag": "en_US", "value": "Leather"}],
    "product_type": [{"value": "CHAIR"}],
    "main_image_id": "img2",
}

NO_ATTR_CHAIR = {
    "item_id": "CHR003",
    "marketplace": "Amazon",
    "country": "US",
    "item_name": [{"language_tag": "en_US", "value": "Basic Mesh Chair"}],
    "product_type": [{"value": "CHAIR"}],
}


def test_import_search_filters_and_foreign_keys(tmp_path: Path) -> None:
    archive = _write_minimal_shards(
        tmp_path / "archive", [FABRIC_CHAIR, LEATHER_CHAIR, NO_ATTR_CHAIR]
    )
    from scripts.import_catalog import import_catalog

    db_path = tmp_path / "catalog.db"
    result = import_catalog(archive, db_path)
    assert result["listings"] == 3

    repo = ABOCatalogRepository(db_path)
    stats = repo.stats()
    assert stats["listings"] == 3
    assert stats["product_types"] == 1

    types = repo.find_product_types("chair")
    assert types == [{"product_type": "CHAIR", "product_count": 3}]

    matches = repo.search("mesh chair", limit=10)
    assert {match["item_id"] for match in matches} == {"CHR001", "CHR003"}

    dim_matches = repo.search("chair", max_dimension_cm=130, limit=10)
    assert "CHR001" in [match["item_id"] for match in dim_matches]

    attrs = {value["attribute"] for value in repo.get_text_values("CHR001")}
    assert {"color", "material", "bullet_point"} <= attrs
    repo.close()


def test_tool_cache_tracks_hits(tmp_path: Path) -> None:
    archive = _write_minimal_shards(
        tmp_path / "archive", [FABRIC_CHAIR, LEATHER_CHAIR, NO_ATTR_CHAIR]
    )
    from scripts.import_catalog import import_catalog

    db_path = tmp_path / "catalog.db"
    import_catalog(archive, db_path)
    repo = ABOCatalogRepository(db_path)
    search_catalog = build_tools(repo)[2]
    clear_cache()
    kwargs = {
        "query": "office chair",
        "product_type": "CHAIR",
        "max_dimension_cm": 0,
        "limit": 10,
    }
    first = search_catalog(**kwargs)
    second = search_catalog(**kwargs)
    assert first == second
    assert json.loads(first)[0]["item_id"] == "CHR002"
    assert cache_stats() == {
        "hits": 1,
        "misses": 1,
        "size": 1,
        "maxsize": 512,
    }
    assert clamp_limit(10_000, default=50, maximum=50) == 50
    assert clamp_limit(0, default=10, maximum=50) == 10
    repo.close()


def test_shopping_agent_wires_tools_in_canonical_order(tmp_path: Path) -> None:
    archive = _write_minimal_shards(
        tmp_path / "archive", [FABRIC_CHAIR, LEATHER_CHAIR, NO_ATTR_CHAIR]
    )
    from scripts.import_catalog import import_catalog

    db_path = tmp_path / "catalog.db"
    import_catalog(archive, db_path)
    repo = ABOCatalogRepository(db_path)

    tracker = CatalogEvidenceTracker()
    catalog_tools = build_tools(repo, catalog_tracker=tracker)
    tools = _build_agent_tools(catalog_tools, tracker=tracker)

    assert [tool.name for tool in tools] == [
        EXTRACT_BRIEF_TOOL,
        "find_product_types",
        "find_brands",
        "search_catalog",
        FINALIZE_RECOMMENDATIONS_TOOL,
    ]
    repo.close()


def test_session_keeps_clarification_answer_in_one_conversation() -> None:
    class FakeAgent:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def create_session(self):
            return object()

        async def run(self, message: str, *, session):
            self.calls.append((message, session))
            if len(self.calls) == 1:
                return type("Response", (), {"text": "Which laptop model do you use?"})()
            return type(
                "Response",
                (),
                {
                    "text": json.dumps(
                        {
                            "kind": "recommendations",
                            "ranked": [],
                            "notes": ["No supported catalog match"],
                            "refinement_chips": [],
                            "dataset_notice": "snapshot",
                        }
                    )
                },
            )()

    async def scenario():
        agent = FakeAgent()
        session = agent.create_session()
        first = await agent.run("I need a laptop charger", session=session)
        second = await agent.run("ThinkPad T14", session=session)
        return agent, session, first, second

    agent, session, first, second = asyncio.run(scenario())
    assert first.text == "Which laptop model do you use?"
    parsed = structured_recommendation_from_response(second)
    assert isinstance(parsed, RecommendationResponse)
    assert parsed.kind == "recommendations"
    assert parsed.ranked == []
    assert [call[1] for call in agent.calls] == [session, session]


def test_finalize_tool_enforces_eligibility_order_and_provenance(tmp_path: Path) -> None:
    archive = _write_minimal_shards(
        tmp_path / "archive", [FABRIC_CHAIR, LEATHER_CHAIR, NO_ATTR_CHAIR]
    )
    from scripts.import_catalog import import_catalog

    db_path = tmp_path / "catalog.db"
    import_catalog(archive, db_path)
    repo = ABOCatalogRepository(db_path)

    tracker = CatalogEvidenceTracker()
    search_catalog = build_tools(repo, catalog_tracker=tracker)[2]
    candidates_payload = json.loads(search_catalog(query="office chair", limit=10))

    finalizer = _make_finalize_tool(tracker)
    catalog_ids = {row["item_id"] for row in candidates_payload}
    accessory_id = next(iter(catalog_ids))
    finalized = finalizer(
        candidates=[
            {
                "item_id": accessory_id,
                "title_en": "Chair Accessory",
                "retrieval_rank": 1,
                "product_type_match": "accessory",
                "has_bullet": True,
                "has_dimensions": False,
                "has_weight": False,
                "has_material": False,
            },
            {
                "item_id": "INVENTED",
                "title_en": "Invented chair",
                "retrieval_rank": 1,
                "product_type_match": "exact_product",
                "has_bullet": True,
                "has_dimensions": False,
                "has_weight": False,
                "has_material": False,
            },
            {
                "item_id": candidates_payload[0]["item_id"],
                "title_en": "Complete Gaming Chair",
                "retrieval_rank": 2,
                "product_type_match": "exact_product",
                "has_bullet": True,
                "has_dimensions": True,
                "has_weight": True,
                "has_material": True,
            },
        ]
    )
    assert [item["item_id"] for item in finalized["candidates"]] == [
        candidates_payload[0]["item_id"]
    ]
    screening = finalized["screening_summary"]
    assert screening["excluded_from_ranking"] == 2
    assert screening["returned_classifications"].get("unknown_to_catalog") == 1
    assert screening["returned_classifications"].get("accessory") == 1

    # enforce_finalized_recommendation expects typed candidates; parse the
    # finalizer output (which is dumped to dicts on the wire) back into
    # FinalizedCandidate before enforcement.
    finalized_typed = [FinalizedCandidate.model_validate(item) for item in finalized["candidates"]]
    protected = enforce_finalized_recommendation(
        {
            "kind": "recommendations",
            "ranked": [
                {"item_id": "COVER", "why_it_fits": ["invented"]},
                {"item_id": "INVENTED", "why_it_fits": ["invented"]},
                {
                    "item_id": candidates_payload[0]["item_id"],
                    "why_it_fits": ["complete chair"],
                },
            ],
            "refinement_chips": [],
        },
        finalized_typed,
    )
    assert isinstance(protected, RecommendationResponse)
    assert [item.item_id for item in protected.ranked] == [
        candidates_payload[0]["item_id"]
    ]
    assert protected.ranked[0].why_it_fits == ["complete chair"]
    repo.close()

    call = Content.from_function_call(
        "call-1", FINALIZE_RECOMMENDATIONS_TOOL, arguments={"candidates": []}
    )
    result = Content.from_function_result("call-1", result=finalized)
    response = AgentResponse(
        messages=[Message("assistant", [call]), Message("tool", [result])]
    )
    extracted = finalized_candidates_from_response(response)
    assert extracted is not None
    assert [item.item_id for item in extracted] == [finalized["candidates"][0]["item_id"]]
    assert all(isinstance(item, FinalizedCandidate) for item in extracted)


def test_multifield_ranking_uses_abo_signals() -> None:
    research = screen_and_rank_candidates(
        {
            "candidates": [
                {
                    "item_id": "LOW",
                    "retrieval_rank": 1,
                    "product_type_match": "exact_product",
                    "has_bullet": False,
                    "has_dimensions": False,
                    "has_weight": False,
                    "has_material": False,
                    "brand_en": None,
                },
                {
                    "item_id": "STRONG",
                    "retrieval_rank": 2,
                    "product_type_match": "exact_product",
                    "has_bullet": True,
                    "has_dimensions": False,
                    "has_weight": True,
                    "has_material": True,
                    "brand_en": "Premium",
                },
            ]
        }
    )
    assert [item.item_id for item in research["candidates"]] == ["STRONG", "LOW"]
    signals = research["candidates"][0].ranking_signals
    assert set(signals) == {
        "text_relevance",
        "bullet_coverage",
        "material_present",
        "brand_present",
        "dimension_present",
    }


def test_extract_json_object_handles_provider_wrappers() -> None:
    import json
    payload = (
        '{"kind":"recommendations","ranked":[],'
        '"notes":[],"refinement_chips":[]}'
    )
    expected = json.loads(payload)
    # 3-backtick fence
    assert json.loads(extract_json_object(f"```json\n{payload}\n```")) == expected
    assert json.loads(extract_json_object(f"```\n{payload}\n```\n```")) == expected
    # Narrative prefix
    assert json.loads(
        extract_json_object(f"Here you go:\n{payload}\nEnjoy.")
    ) == expected
    # Reasoning-tag prefix
    assert json.loads(
        extract_json_object(f"<think>\nthinking\n</think>\n{payload}")
    ) == expected
    # Plain JSON passes through unchanged
    assert json.loads(extract_json_object(payload)) == expected


def test_structured_recommendation_from_response_validates_against_schema() -> None:
    payload = (
        '{"kind":"recommendations","ranked":[],'
        '"notes":[],"refinement_chips":[]}'
    )
    fake = type("Response", (), {"text": f"```json\n{payload}\n```"})()
    parsed = structured_recommendation_from_response(fake)
    assert isinstance(parsed, RecommendationResponse)
    assert parsed.kind == "recommendations"


def test_human_readable_product_output() -> None:
    rendered = _format_product(
        {
            "rank": 1,
            "title_en": "Travel Headphones",
            "brand_en": "SoundPro",
            "why_it_fits": ["Noise cancelling"],
            "trade_offs": ["Comfort is not verified"],
            "product_url": "https://example.com/HP1",
        }
    )
    assert "1. Travel Headphones" in rendered
    assert "SoundPro" in rendered
    assert "+ Noise cancelling" in rendered
    assert "- Comfort is not verified" in rendered
