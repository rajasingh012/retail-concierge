"""Tests for the JSON extraction helper used by the recommendation fallback.

The harness payloads are the source of truth here, not synthetic cases.
"""
from __future__ import annotations

import json

from domain.recommendation import (
    RecommendationResponse,
    extract_json_object,
)


def test_extract_json_object_handles_fenced_with_braces() -> None:
    """Standard ```json { ... } ``` fence extracts cleanly."""
    text = (
        "```json\n"
        '{"kind": "recommendations", "ranked": [], "refinement_chips": []}\n'
        "```"
    )
    out = extract_json_object(text)
    parsed = json.loads(out)
    assert parsed["kind"] == "recommendations"
    assert parsed["ranked"] == []


def test_extract_json_object_handles_missing_outer_brace() -> None:
    """The exact Q1 failure mode: fence-strip ate the opening brace.
    The model emitted the JSON inside a ```json fence with the opening
    brace on the same line as the fence opener, leaving ``kind":"..."``
    and the balanced-brace scan returned the inner ranked[0] object.
    """
    text = (
        "```json\n"
        'kind":"recommendations","ranked":[{"rank":1,"item_id":"B07MC2NTJX",'
        '"title_en":"AmazonBasics Wireless Bluetooth Fitness Headphones Earbuds",'
        '"brand_en":"AmazonBasics","product_type":"HEADPHONES",'
        '"product_url":"https://www.Amazon/dp/B07MC2NTJX",'
        '"why_it_fits":["True wireless earbuds with Bluetooth"],'
        '"trade_offs":["Cataloged on Amazon SA"]}],'
        '"assumptions":[],"notes":[],"refinement_chips":[]}\n'
        "```"
    )
    out = extract_json_object(text)
    parsed = json.loads(out)
    assert parsed["kind"] == "recommendations"
    assert len(parsed["ranked"]) == 1
    assert parsed["ranked"][0]["item_id"] == "B07MC2NTJX"


def test_extract_json_object_validates_against_recommendation_schema() -> None:
    """The fix must produce input that RecommendationResponse can validate."""
    text = (
        "```json\n"
        'kind":"recommendations","ranked":[{"rank":1,"item_id":"B07H7RPJ7P",'
        '"title_en":"UMI water bottle","brand_en":"UMI","product_type":"THERMOS",'
        '"product_url":"https://www.Amazon/dp/B07H7RPJ7P","why_it_fits":[],'
        '"trade_offs":[]}],"assumptions":[],"notes":[],"refinement_chips":[]}\n'
        "```"
    )
    out = extract_json_object(text)
    response = RecommendationResponse.model_validate_json(out)
    assert response.kind == "recommendations"
    assert response.ranked[0].item_id == "B07H7RPJ7P"


def test_extract_json_object_handles_prose_around_json() -> None:
    """Plain prose with the JSON object buried mid-text."""
    text = (
        "Here is the recommendation:\n"
        '{"kind":"recommendations","ranked":[],"refinement_chips":[]}\n'
        "Let me know if you want to refine."
    )
    out = extract_json_object(text)
    parsed = json.loads(out)
    assert parsed["kind"] == "recommendations"


def test_extract_json_object_returns_text_on_unparseable_input() -> None:
    """Non-JSON input falls back to the original text."""
    text = "This is not JSON at all."
    out = extract_json_object(text)
    assert out == text
