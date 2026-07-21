from __future__ import annotations

import asyncio
import csv
import json
import sqlite3
from pathlib import Path

from infrastructure.agent_tools import build_tools, cache_stats, clear_cache
from infrastructure.database import ProductCatalogRepository
from main import _format_product
from scripts.import_catalog import import_catalog
from use_cases.collaboration import (
    RefinementChip,
    apply_refinement,
    parse_json_object,
    parse_refinement_chips,
    run_collaboration,
)


def _write_dataset(root: Path) -> tuple[Path, Path]:
    categories = root / "categories.csv"
    products = root / "products.csv"
    with categories.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "category_name"])
        writer.writerow([1, "Office Chairs"])
        writer.writerow([2, "Headphones"])
    with products.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "asin", "title", "imgUrl", "productURL", "stars", "reviews",
            "price", "listPrice", "category_id", "isBestSeller",
            "boughtInLastMonth",
        ])
        writer.writerow([
            "CHAIR1", "Ergonomic Mesh Office Chair Lumbar Support", "img1",
            "https://example.com/CHAIR1", 4.6, 1200, 149.99, 199.99, 1,
            "True", 500,
        ])
        writer.writerow([
            "CHAIR2", "Leather Executive Office Chair", "img2",
            "https://example.com/CHAIR2", 4.1, 300, 189.99, 0, 1,
            "False", 80,
        ])
        writer.writerow([
            "CHAIR3", "Mesh Office Chair", "img3",
            "https://example.com/CHAIR3", 4.9, 10, 0, 0, 1,
            "False", 5,
        ])
    return products, categories


def test_import_search_filters_and_foreign_keys(tmp_path: Path) -> None:
    products, categories = _write_dataset(tmp_path)
    database = tmp_path / "catalog.db"
    result = import_catalog(products, categories, database)
    assert result["products"] == 3
    assert result["categories"] == 2

    repository = ProductCatalogRepository(database)
    assert repository.stats() == {"products": 3, "categories": 2}
    assert repository.find_categories("chair")[0]["category_id"] == 1
    matches = repository.search(
        "ergonomic mesh", category_id=1, max_price=160, min_stars=4.5
    )
    assert [product.asin for product in matches] == ["CHAIR1"]
    product = repository.get_by_asin("CHAIR1")
    assert product is not None
    assert product.is_best_seller is True
    repository.close()

    conn = sqlite3.connect(database)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute(
            """
            INSERT INTO products(
                asin,title,image_url,product_url,stars,review_count,price,
                list_price,category_id,is_best_seller,bought_last_month
            ) VALUES ('BAD','Bad','x','x',4,0,1,0,999,0,0)
            """
        )
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("category foreign key was not enforced")
    finally:
        conn.close()


def test_tool_cache_tracks_hits(tmp_path: Path) -> None:
    products, categories = _write_dataset(tmp_path)
    database = tmp_path / "catalog.db"
    import_catalog(products, categories, database)
    repository = ProductCatalogRepository(database)
    search_catalog = build_tools(repository)[1]
    clear_cache()
    kwargs = {
        "query": "ergonomic chair",
        "category_id": 1,
        "max_price": 200,
        "min_stars": 4,
        "bestseller_only": False,
        "limit": 5,
    }
    first = search_catalog(**kwargs)
    second = search_catalog(**kwargs)
    assert first == second
    assert json.loads(first)[0]["asin"] == "CHAIR1"
    assert cache_stats() == {"hits": 1, "misses": 1, "size": 1, "maxsize": 512}
    repository.close()


def test_blocking_ambiguity_requests_clarification() -> None:
    responses = {
        "discovery": iter([
            '{"complete": false, "question": "Which laptop model do you use?"}',
            '{"complete": true, "brief": {"intent": "laptop charger", '
            '"search_terms": "ThinkPad USB-C charger", "category_hint": "charger", '
            '"budget_max": 0, "minimum_stars": 0, "bestseller_only": false, '
            '"must_have": ["ThinkPad compatible"], "nice_to_have": [], '
            '"target_use": "ThinkPad T14", "assumptions": []}}',
        ]),
        "research": iter([
            '{"brief": {}, "searches": ["ThinkPad USB-C charger"], '
            '"candidates": [{"asin": "CHARGER1"}], '
            '"dataset_notice": "snapshot"}'
        ]),
        "critic": iter([
            '{"ranked": [{"asin": "CHARGER1"}], "critic_notes": [], '
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
    assert result.research["candidates"][0]["asin"] == "CHARGER1"
    assert result.recommendation["ranked"][0]["asin"] == "CHARGER1"
    assert result.refinement_chips == (
        RefinementChip("Prefer compact", "Prioritize a compact charger"),
    )


def test_complete_brief_skips_clarification() -> None:
    responses = {
        "discovery": iter([
            '{"complete": true, "brief": {"intent": "headphones", '
            '"search_terms": "noise cancelling headphones", '
            '"category_hint": "headphones", "budget_max": 250, '
            '"minimum_stars": 4, "bestseller_only": false, '
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
            "Noise-cancelling headphones under $250 for commuting",
            unexpected_clarification,
            run_agent=fake_run,
        )
    )
    assert result.clarifications_requested == 0
    assert result.brief["budget_max"] == 250


def test_nonblocking_ambiguity_proceeds_with_assumptions_and_refinements() -> None:
    responses = {
        "discovery": iter([
            '{"complete": true, "brief": {"intent": "travel headphones", '
            '"search_terms": "noise cancelling headphones", '
            '"category_hint": "headphones", "budget_max": 0, '
            '"minimum_stars": 0, "bestseller_only": false, '
            '"must_have": [], "nice_to_have": ["travel friendly"], '
            '"target_use": "travel", '
            '"assumptions": ["Over-ear models are acceptable"]}}'
        ]),
        "research": iter([
            '{"brief": {}, "searches": ["noise cancelling headphones"], '
            '"candidates": [{"asin": "HP1"}], "dataset_notice": "snapshot"}'
        ]),
        "critic": iter([
            '{"ranked": [{"asin": "HP1"}], "critic_notes": [], '
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
            "discovery",
            "research",
            "critic",
            "I need headphones for travel",
            unexpected_clarification,
            run_agent=fake_run,
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
        "title": "Travel Headphones",
        "dataset_price": 199.99,
        "dataset_stars": 4.6,
        "why_it_fits": ["Within budget"],
        "trade_offs": ["Comfort is not verified"],
        "product_url": "https://example.com/HP1",
    })
    assert "1. Travel Headphones" in rendered
    assert "Dataset price: $199.99" in rendered
    assert "+ Within budget" in rendered
    assert "- Comfort is not verified" in rendered


def test_parse_json_object_accepts_fence_and_rejects_array() -> None:
    assert parse_json_object("```json\n{\"ok\": true}\n```", "test") == {"ok": True}
    try:
        parse_json_object("[]", "test")
    except ValueError:
        pass
    else:
        raise AssertionError("non-object JSON should fail")
