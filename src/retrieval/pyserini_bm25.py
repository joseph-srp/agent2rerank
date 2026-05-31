from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENAI_API_KEY", "sk-pyserini-lucene-only")

from pyserini.search.lucene import LuceneSearcher


class BM25Retriever:
    def __init__(self, index_dir: str | Path, k1: float = 0.9, b: float = 0.4):
        self.searcher = LuceneSearcher(str(index_dir))
        self.searcher.set_bm25(k1, b)

    def search(self, query: str, top_k: int = 100) -> list[dict[str, Any]]:
        hits = self.searcher.search(query, k=top_k)
        results: list[dict[str, Any]] = []
        for rank, hit in enumerate(hits, start=1):
            raw = self.searcher.doc(hit.docid).raw()
            record = json.loads(raw) if raw else {"product_id": hit.docid, "product_text": ""}
            results.append(
                {
                    "rank": rank,
                    "product_id": record.get("product_id") or record.get("id") or hit.docid,
                    "bm25_score": float(hit.score),
                    "product_text": record.get("product_text") or record.get("contents", ""),
                }
            )
        return results


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_bm25(index_dir: Path, queries_path: Path, output_path: Path, top_k: int, k1: float, b: float) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    retriever = BM25Retriever(index_dir=index_dir, k1=k1, b=b)
    query_count = 0
    result_count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for query in iter_jsonl(queries_path):
            results = retriever.search(query["query"], top_k=top_k)
            query_count += 1
            result_count += len(results)
            out.write(
                json.dumps(
                    {"query_id": query["query_id"], "query": query["query"], "bm25_top100": results},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    return {"queries": query_count, "results": result_count, "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pyserini BM25 retrieval for ESCI queries.")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    args = parser.parse_args()
    stats = run_bm25(args.index_dir, args.queries, args.output, args.top_k, args.k1, args.b)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()