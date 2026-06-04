from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.agent.tools import iter_jsonl

LABEL_GAINS = {"E": 3.0, "S": 2.0, "C": 1.0, "I": 0.0}


def dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))


def load_qrels(path: str | Path) -> dict[str, dict[str, str]]:
    qrels: dict[str, dict[str, str]] = defaultdict(dict)
    for record in iter_jsonl(path):
        qrels[str(record["query_id"])][record["product_id"]] = record["esci_label"]
    return dict(qrels)


def evaluate_rerank_records(records: list[dict[str, Any]], qrels: dict[str, dict[str, str]], *, k: int = 10) -> dict[str, float]:
    ndcg_values: list[float] = []
    mrr_values: list[float] = []
    hit_values: list[float] = []
    recall_values: list[float] = []

    for record in records:
        query_id = str(record["query_id"])
        query_qrels = qrels.get(query_id, {})
        if not query_qrels:
            continue
        ranked = record.get("reranked_top100", [])[:k]
        gains = [LABEL_GAINS.get(query_qrels.get(hit["product_id"], "I"), 0.0) for hit in ranked]
        ideal = sorted((LABEL_GAINS.get(label, 0.0) for label in query_qrels.values()), reverse=True)[:k]
        ideal_dcg = dcg(ideal)
        ndcg_values.append(dcg(gains) / ideal_dcg if ideal_dcg else 0.0)

        exact_products = {product_id for product_id, label in query_qrels.items() if label == "E"}
        if not exact_products:
            continue
        first_exact_rank = None
        exact_hits = 0
        for rank, hit in enumerate(ranked, start=1):
            if hit["product_id"] in exact_products:
                exact_hits += 1
                if first_exact_rank is None:
                    first_exact_rank = rank
        mrr_values.append(1.0 / first_exact_rank if first_exact_rank else 0.0)
        hit_values.append(1.0 if first_exact_rank else 0.0)
        recall_values.append(exact_hits / len(exact_products))

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        f"ndcg_at_{k}": mean(ndcg_values),
        f"mrr_at_{k}_exact": mean(mrr_values),
        f"hit_at_{k}_exact": mean(hit_values),
        f"recall_at_{k}_exact": mean(recall_values),
    }


def evaluate_rerank_file(run_path: str | Path, qrels_path: str | Path, output_path: str | Path | None = None, *, k: int = 10) -> dict[str, Any]:
    qrels = load_qrels(qrels_path)
    records = list(iter_jsonl(run_path))
    result = {
        "input_run": str(run_path),
        "qrels": str(qrels_path),
        "queries": len(records),
        "metrics": evaluate_rerank_records(records, qrels, k=k),
    }
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
