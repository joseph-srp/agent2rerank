from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def iter_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def compact_product_snippet(product_text: str, max_chars: int = 700) -> str:
    text = " ".join((product_text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


class SearchTool:
    name = "search"

    def __init__(self, retriever, top_k: int = 10, snippet_chars: int = 700):
        self.retriever = retriever
        self.top_k = top_k
        self.snippet_chars = snippet_chars

    @classmethod
    def from_index(cls, index_dir: str | Path, *, top_k: int = 10, k1: float = 0.9, b: float = 0.4) -> "SearchTool":
        from src.retrieval.pyserini_bm25 import BM25Retriever

        return cls(BM25Retriever(index_dir=index_dir, k1=k1, b=b), top_k=top_k)

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        results = self.retriever.search(query, top_k=top_k or self.top_k)
        return [
            {
                "rank": item["rank"],
                "product_id": item["product_id"],
                "bm25_score": item["bm25_score"],
                "snippet": compact_product_snippet(item.get("product_text", ""), self.snippet_chars),
            }
            for item in results
        ]

    def __call__(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.search(query=query, top_k=top_k)


class DocumentTool:
    name = "get_document"

    def __init__(self, products_path: str | Path):
        self.products_path = Path(products_path)
        self.offsets: dict[str, int] = {}
        with self.products_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                record = json.loads(line)
                self.offsets[record["product_id"]] = offset

    def get_document(self, product_id: str) -> dict[str, Any]:
        offset = self.offsets.get(product_id)
        if offset is None:
            raise KeyError(f"product_id not found: {product_id}")
        with self.products_path.open("rb") as handle:
            handle.seek(offset)
            record = json.loads(handle.readline())
        fields = [
            "product_id",
            "product_title",
            "product_brand",
            "product_color",
            "product_bullet_point",
            "product_description",
            "product_text",
        ]
        return {field: record.get(field, "") for field in fields}

    def __call__(self, product_id: str) -> dict[str, Any]:
        return self.get_document(product_id=product_id)


def openai_tool_specs(max_search_top_k: int = 20) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the local BM25 product index for products matching a shopping query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": max_search_top_k},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_document",
                "description": "Open a full product record by product_id.",
                "parameters": {
                    "type": "object",
                    "properties": {"product_id": {"type": "string"}},
                    "required": ["product_id"],
                },
            },
        },
    ]
