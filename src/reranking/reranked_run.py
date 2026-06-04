from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agent.tools import iter_jsonl


class RerankedRunRetriever:
    def __init__(self, run_path: str | Path):
        self.run_path = Path(run_path)
        self.by_query: dict[str, dict[str, Any]] = {}
        self.by_query_id: dict[str, dict[str, Any]] = {}
        for record in iter_jsonl(self.run_path):
            self.by_query[str(record["query"])] = record
            self.by_query_id[str(record["query_id"])] = record

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        record = self.by_query.get(query) or self.by_query_id.get(query)
        if not record:
            return []
        results: list[dict[str, Any]] = []
        for hit in record.get("reranked_top100", [])[:top_k]:
            results.append(
                {
                    "rank": int(hit["rank"]),
                    "product_id": hit["product_id"],
                    "bm25_score": float(hit.get("bm25_score", 0.0)),
                    "rerank_score": float(hit.get("rerank_score", 0.0)),
                    "product_text": hit.get("product_text", ""),
                }
            )
        return results


class SessionRerankedRunRetriever(RerankedRunRetriever):
    def __init__(self, run_path: str | Path):
        super().__init__(run_path)
        self.current_query_id: str | None = None
        self.current_query: str | None = None

    def set_session(self, query_id: str, query: str) -> None:
        self.current_query_id = str(query_id)
        self.current_query = query

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        record = None
        if self.current_query_id:
            record = self.by_query_id.get(self.current_query_id)
        if record is None and self.current_query:
            record = self.by_query.get(self.current_query)
        if record is None:
            record = self.by_query.get(query) or self.by_query_id.get(query)
        if not record:
            return []
        results: list[dict[str, Any]] = []
        for hit in record.get("reranked_top100", [])[:top_k]:
            results.append(
                {
                    "rank": int(hit["rank"]),
                    "product_id": hit["product_id"],
                    "bm25_score": float(hit.get("bm25_score", 0.0)),
                    "rerank_score": float(hit.get("rerank_score", 0.0)),
                    "product_text": hit.get("product_text", ""),
                }
            )
        return results
