from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RerankHit:
    rank: int
    product_id: str
    rerank_score: float
    bm25_rank: int
    bm25_score: float
    product_text: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RerankRecord:
    query_id: str
    query: str
    reranked_top100: list[RerankHit]

    def to_json(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "reranked_top100": [hit.to_json() for hit in self.reranked_top100],
        }
