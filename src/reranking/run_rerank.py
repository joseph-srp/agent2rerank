from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from src.agent.tools import iter_jsonl
from src.reranking.schemas import RerankHit, RerankRecord


class HitScorer(Protocol):
    def score_hits(self, query: str, hits: list[dict[str, Any]], batch_size: int = 8) -> list[float]:
        ...


def rerank_record(record: dict[str, Any], scorer: HitScorer, *, candidate_top_k: int = 100, batch_size: int = 8) -> RerankRecord:
    query = record["query"]
    hits = list(record.get("bm25_top100", []))[:candidate_top_k]
    scores = scorer.score_hits(query, hits, batch_size=batch_size)
    if len(scores) != len(hits):
        raise ValueError(f"scorer returned {len(scores)} scores for {len(hits)} hits")
    scored = []
    for hit, score in zip(hits, scores):
        scored.append(
            {
                "product_id": hit["product_id"],
                "rerank_score": float(score),
                "bm25_rank": int(hit.get("rank", 0)),
                "bm25_score": float(hit.get("bm25_score", 0.0)),
                "product_text": hit.get("product_text", ""),
            }
        )
    scored.sort(key=lambda item: (-item["rerank_score"], item["bm25_rank"]))
    reranked = [
        RerankHit(
            rank=rank,
            product_id=item["product_id"],
            rerank_score=item["rerank_score"],
            bm25_rank=item["bm25_rank"],
            bm25_score=item["bm25_score"],
            product_text=item["product_text"],
        )
        for rank, item in enumerate(scored, start=1)
    ]
    return RerankRecord(query_id=str(record["query_id"]), query=query, reranked_top100=reranked)


def run_rerank_file(
    bm25_run_path: str | Path,
    output_path: str | Path,
    scorer: HitScorer,
    *,
    candidate_top_k: int = 100,
    batch_size: int = 8,
    limit: int | None = None,
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    queries = 0
    hits = 0
    with output.open("w", encoding="utf-8") as out:
        for record in iter_jsonl(bm25_run_path):
            reranked = rerank_record(record, scorer, candidate_top_k=candidate_top_k, batch_size=batch_size)
            queries += 1
            hits += len(reranked.reranked_top100)
            out.write(json.dumps(reranked.to_json(), ensure_ascii=False, separators=(",", ":")) + "\n")
            if limit and queries >= limit:
                break
    return {"input": str(bm25_run_path), "output": str(output), "queries": queries, "hits": hits}
