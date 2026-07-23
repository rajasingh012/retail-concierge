"""Integration test for MAF session history and the deterministic finalizer tool."""
from __future__ import annotations

import asyncio
import gzip
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agent_framework import ChatResponse, Content, Message
from agent_framework._clients import BaseChatClient
from agent_framework._tools import FunctionInvocationLayer

from use_cases.shopping_agent import (
    FINALIZE_RECOMMENDATIONS_TOOL,
    build_shopping_agent,
    finalized_candidates_from_response,
)
from infrastructure.agent_tools import build_tools as build_catalog_tools
from infrastructure.database import ABOCatalogRepository


CHARGER_RAW = {
    "item_id": "CHARGER1",
    "marketplace": "Amazon",
    "country": "US",
    "item_name": [
        {"language_tag": "en_US", "value": "ThinkPad USB-C 65W Charger"}
    ],
    "brand": [{"language_tag": "en_US", "value": "Lenovo"}],
    "product_type": [{"value": "POWER_ADAPTER"}],
    "main_image_id": "img",
}


def _write_minimal_shards(root: Path, rows: list[dict]) -> Path:
    shard_dir = root / "listings" / "metadata"
    shard_dir.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    (shard_dir / "listings_0.json.gz").write_bytes(gzip.compress(lines.encode("utf-8")))
    return root


class ScriptedChatClient(FunctionInvocationLayer, BaseChatClient):
    """Small MAF-compatible client that scripts clarification then tool use."""

    STORES_BY_DEFAULT = False

    def __init__(self, charger_id: str) -> None:
        super().__init__()
        self.calls: list[list[Message]] = []
        self.charger_id = charger_id

    async def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **_kwargs: Any,
    ) -> ChatResponse[Any]:
        assert stream is False
        self.calls.append(list(messages))
        call_number = len(self.calls)
        if call_number == 1:
            return ChatResponse(
                messages=[Message("assistant", ["Which laptop model do you use?"])]
            )
        if call_number == 2:
            return ChatResponse(
                messages=[
                    Message(
                        "assistant",
                        [
                            Content.from_function_call(
                                "finalize-1",
                                FINALIZE_RECOMMENDATIONS_TOOL,
                                arguments={
                                    "candidates": [
                                        {
                                            "item_id": self.charger_id,
                                            "title_en": "ThinkPad USB-C 65W Charger",
                                            "retrieval_rank": 1,
                                            "product_type_match": "exact_product",
                                            "has_bullet": True,
                                            "has_dimensions": False,
                                            "has_weight": True,
                                            "has_material": False,
                                        }
                                    ]
                                },
                            )
                        ],
                    )
                ]
            )
        return ChatResponse(
            messages=[
                Message(
                    "assistant",
                    [
                        json.dumps(
                            {
                                "kind": "recommendations",
                                "ranked": [{"item_id": self.charger_id}],
                                "notes": [],
                                "refinement_chips": [],
                            }
                        )
                    ],
                )
            ]
        )


def test_real_agent_run_reuses_session_and_enforces_provenance(tmp_path: Path) -> None:
    archive = _write_minimal_shards(tmp_path / "archive", [CHARGER_RAW])
    from scripts.import_catalog import import_catalog

    db_path = tmp_path / "catalog.db"
    import_catalog(archive, db_path)
    repo = ABOCatalogRepository(db_path)

    tracker = type("Tracker", (), {})  # placeholder, replaced below
    from use_cases.shopping_agent import CatalogEvidenceTracker

    tracker = CatalogEvidenceTracker()
    catalog_tools = build_catalog_tools(repo, catalog_tracker=tracker)
    search_catalog = next(
        tool for tool in catalog_tools if getattr(tool, "name", "") == "search_catalog"
    )

    search_catalog(query="ThinkPad charger", product_type="POWER_ADAPTER", limit=10)

    client = ScriptedChatClient("CHARGER1")
    agent = build_shopping_agent(client, catalog_tools, tracker=tracker)
    session = agent.create_session()

    first = asyncio.run(agent.run("I need a laptop charger", session=session))
    second = asyncio.run(agent.run("ThinkPad T14", session=session))

    assert first.text == "Which laptop model do you use?"
    second_call_text = " ".join(message.text for message in client.calls[1])
    assert "I need a laptop charger" in second_call_text
    assert "Which laptop model do you use?" in second_call_text
    assert "ThinkPad T14" in second_call_text
    finalized = finalized_candidates_from_response(second)
    assert finalized is not None
    assert [candidate.item_id for candidate in finalized] == ["CHARGER1"]
    repo.close()


def test_invented_ids_are_dropped_by_finalizer(tmp_path: Path) -> None:
    archive = _write_minimal_shards(tmp_path / "archive", [CHARGER_RAW])
    from scripts.import_catalog import import_catalog

    db_path = tmp_path / "catalog.db"
    import_catalog(archive, db_path)
    repo = ABOCatalogRepository(db_path)

    from use_cases.shopping_agent import CatalogEvidenceTracker

    tracker = CatalogEvidenceTracker()
    catalog_tools = build_catalog_tools(repo, catalog_tracker=tracker)
    search_catalog = next(
        tool for tool in catalog_tools if getattr(tool, "name", "") == "search_catalog"
    )
    search_catalog(query="ThinkPad charger", product_type="POWER_ADAPTER", limit=10)

    finalizer = next(
        tool
        for tool in build_shopping_agent(
            type("C", (), {})(), catalog_tools, tracker=tracker
        ).default_options["tools"]
        if getattr(tool, "name", "") == FINALIZE_RECOMMENDATIONS_TOOL
    )
    finalized = finalizer(
        candidates=[
            {
                "item_id": "CHARGER1",
                "title_en": "ThinkPad USB-C 65W Charger",
                "retrieval_rank": 1,
                "product_type_match": "exact_product",
                "has_bullet": True,
                "has_dimensions": False,
                "has_weight": True,
                "has_material": False,
            },
            {
                "item_id": "INVENTED-ID",
                "title_en": "Hallucinated charger",
                "retrieval_rank": 1,
                "product_type_match": "exact_product",
                "has_bullet": True,
                "has_dimensions": False,
                "has_weight": False,
                "has_material": False,
            },
        ]
    )
    assert [item["item_id"] for item in finalized["candidates"]] == ["CHARGER1"]
    assert finalized["screening_summary"]["returned_classifications"].get(
        "unknown_to_catalog"
    ) == 1
    repo.close()
