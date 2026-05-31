from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.output_parser import TrajectoryParseError, parse_final_trajectory
from src.agent.schemas import QueryRecord
from src.agent.tools import DocumentTool, SearchTool
from src.signal.pair_builder import PairBuilder


class FakeRetriever:
    def search(self, query: str, top_k: int = 100):
        return [
            {"rank": 1, "product_id": "p1", "bm25_score": 1.5, "product_text": "Title: Good product Brand: A"},
            {"rank": 2, "product_id": "p2", "bm25_score": 1.0, "product_text": "Title: Bad product Brand: B"},
        ][:top_k]


def write_products(path: Path) -> None:
    rows = [
        {"product_id": "p1", "product_title": "Good", "product_brand": "A", "product_color": "", "product_bullet_point": "", "product_description": "", "product_text": "Title: Good"},
        {"product_id": "p2", "product_title": "Bad", "product_brand": "B", "product_color": "", "product_bullet_point": "", "product_description": "", "product_text": "Title: Bad"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_search_tool_returns_ranked_snippets():
    results = SearchTool(FakeRetriever(), top_k=2).search("good")
    assert len(results) == 2
    assert results[0]["product_id"] == "p1"
    assert set(results[0]) == {"rank", "product_id", "bm25_score", "snippet"}


def test_document_tool_returns_full_metadata(tmp_path: Path):
    products_path = tmp_path / "products.jsonl"
    write_products(products_path)
    doc = DocumentTool(products_path).get_document("p1")
    assert doc["product_id"] == "p1"
    assert doc["product_text"] == "Title: Good"
    with pytest.raises(KeyError):
        DocumentTool(products_path).get_document("missing")


def test_trajectory_parser_validates_required_fields():
    valid = json.dumps({
        "query_id": "q1",
        "original_query": "case",
        "agent_search_query": "case",
        "opened_products": ["p1", "p2"],
        "rejected_products": [{"product_id": "p2", "reason": "wrong type"}],
        "selected_product_id": "p1",
        "final_answer": "p1 is best",
    })
    parsed = parse_final_trajectory(valid, query_id="q1", original_query="case")
    assert parsed.selected_product_id == "p1"

    with pytest.raises(TrajectoryParseError):
        parse_final_trajectory('{"opened_products": []}', query_id="q1", original_query="case")
    with pytest.raises(TrajectoryParseError):
        parse_final_trajectory(json.dumps({
            "opened_products": ["p1"],
            "rejected_products": ["p2"],
            "selected_product_id": "p1",
            "final_answer": "x",
        }), query_id="q1", original_query="case")


def test_pair_builder_filters_and_skips_invalid_pairs(tmp_path: Path):
    products_path = tmp_path / "products.jsonl"
    write_products(products_path)
    builder = PairBuilder(DocumentTool(products_path))
    trajectory = {
        "query_id": "q1",
        "original_query": "good",
        "agent_search_query": "good product",
        "selected_product_id": "p1",
        "rejected_products": [
            {"product_id": "p2", "reason": "less relevant"},
            {"product_id": "p1", "reason": "self pair"},
            {"product_id": "missing", "reason": "missing"},
        ],
    }
    success = {"verifier_success": True, "verifier_score": 3, "verifier_confidence": 0.7}
    failed = {"verifier_success": False, "verifier_score": 5, "verifier_confidence": 1.0}
    pairs = builder.build_pairs(trajectory, success)
    assert len(pairs) == 1
    assert pairs[0]["positive_product_id"] == "p1"
    assert pairs[0]["negative_product_id"] == "p2"
    assert builder.build_pairs(trajectory, failed) == []
