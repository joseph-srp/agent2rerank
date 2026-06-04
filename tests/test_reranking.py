from __future__ import annotations

import json
from pathlib import Path

from src.reranking.metrics import evaluate_rerank_records
from src.reranking.qwen3_reranker import format_qwen3_reranker_input
from src.reranking.reranked_run import RerankedRunRetriever
from src.reranking.run_rerank import rerank_record, run_rerank_file


class FixedScorer:
    def score_hits(self, query, hits, batch_size=8):
        return [0.2, 0.9, 0.2]


def test_qwen3_formatter_contains_instruction_query_and_document():
    text = format_qwen3_reranker_input("iphone case", "Title: blue case", "Judge relevance.")
    assert "Judge relevance." in text
    assert "<Query>: iphone case" in text
    assert "<Document>: Title: blue case" in text


def test_rerank_record_sorts_by_score_then_bm25_rank():
    record = {
        "query_id": "q1",
        "query": "case",
        "bm25_top100": [
            {"rank": 1, "product_id": "p1", "bm25_score": 10.0, "product_text": "one"},
            {"rank": 2, "product_id": "p2", "bm25_score": 9.0, "product_text": "two"},
            {"rank": 3, "product_id": "p3", "bm25_score": 8.0, "product_text": "three"},
        ],
    }
    output = rerank_record(record, FixedScorer(), candidate_top_k=3)
    hits = output.to_json()["reranked_top100"]
    assert [hit["product_id"] for hit in hits] == ["p2", "p1", "p3"]
    assert [hit["rank"] for hit in hits] == [1, 2, 3]
    assert hits[0]["bm25_rank"] == 2


def test_evaluate_rerank_records_tiny_fixture():
    records = [
        {"query_id": "q1", "reranked_top100": [{"product_id": "p2"}, {"product_id": "p1"}]},
        {"query_id": "q2", "reranked_top100": [{"product_id": "p4"}]},
    ]
    qrels = {
        "q1": {"p1": "E", "p2": "I"},
        "q2": {"p4": "E", "p5": "E"},
    }
    metrics = evaluate_rerank_records(records, qrels, k=2)
    assert metrics["hit_at_2_exact"] == 1.0
    assert metrics["mrr_at_2_exact"] == 0.75
    assert metrics["recall_at_2_exact"] == 0.75
    assert 0.0 < metrics["ndcg_at_2"] <= 1.0


def test_run_rerank_file_writes_jsonl(tmp_path: Path):
    input_path = tmp_path / "bm25.jsonl"
    output_path = tmp_path / "rerank.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "case",
                "bm25_top100": [
                    {"rank": 1, "product_id": "p1", "bm25_score": 1.0, "product_text": "one"},
                    {"rank": 2, "product_id": "p2", "bm25_score": 2.0, "product_text": "two"},
                    {"rank": 3, "product_id": "p3", "bm25_score": 3.0, "product_text": "three"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stats = run_rerank_file(input_path, output_path, FixedScorer(), candidate_top_k=3)
    assert stats["queries"] == 1
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["reranked_top100"][0]["product_id"] == "p2"



def test_reranked_run_retriever_returns_rerank_scores(tmp_path: Path):
    run_path = tmp_path / "rerank.jsonl"
    run_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "case",
                "reranked_top100": [
                    {"rank": 1, "product_id": "p2", "rerank_score": 0.9, "bm25_rank": 2, "bm25_score": 2.0, "product_text": "two"}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    results = RerankedRunRetriever(run_path).search("case", top_k=1)
    assert results == [{"rank": 1, "product_id": "p2", "bm25_score": 2.0, "rerank_score": 0.9, "product_text": "two"}]



def test_session_reranked_run_retriever_prefers_query_id(tmp_path: Path):
    from src.reranking.reranked_run import SessionRerankedRunRetriever

    run_path = tmp_path / "rerank.jsonl"
    records = [
        {"query_id": "q1", "query": "original", "reranked_top100": [{"rank": 1, "product_id": "p1", "rerank_score": 0.8, "bm25_rank": 1, "bm25_score": 1.0, "product_text": "one"}]},
        {"query_id": "q2", "query": "rewritten", "reranked_top100": [{"rank": 1, "product_id": "p2", "rerank_score": 0.9, "bm25_rank": 1, "bm25_score": 1.0, "product_text": "two"}]},
    ]
    run_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    retriever = SessionRerankedRunRetriever(run_path)
    retriever.set_session("q1", "original")
    assert retriever.search("rewritten", top_k=1)[0]["product_id"] == "p1"
