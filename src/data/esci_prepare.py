from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")

PRODUCT_COLUMNS = [
    "product_id",
    "product_locale",
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
]
QUERY_COLUMNS = ["query_id", "query", "split"]
READ_COLUMNS = QUERY_COLUMNS + ["esci_label"] + PRODUCT_COLUMNS


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("\u2028", " ").replace("\u2029", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def build_product_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    field_map = [
        ("Title", "product_title"),
        ("Brand", "product_brand"),
        ("Color", "product_color"),
        ("Bullet points", "product_bullet_point"),
        ("Description", "product_description"),
    ]
    for label, key in field_map:
        text = clean_text(record.get(key))
        if text:
            parts.append(f"{label}: {text}")
    return " ".join(parts)


def iter_batches(paths: Iterable[Path], batch_size: int):
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        columns = [col for col in READ_COLUMNS if col in available]
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            yield path, batch.to_pydict()


def write_jsonl_record(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare_esci(raw_dir: Path, output_dir: Path, batch_size: int = 25_000) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lucene_docs_dir = output_dir / "lucene_docs"
    lucene_docs_dir.mkdir(parents=True, exist_ok=True)

    parquet_paths = sorted(raw_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {raw_dir}")

    products_path = output_dir / "products.jsonl"
    qrels_paths = {
        "train": output_dir / "qrels_train.jsonl",
        "test": output_dir / "qrels_test.jsonl",
    }
    queries_paths = {
        "train": output_dir / "queries_train.jsonl",
        "test": output_dir / "queries_test.jsonl",
    }
    lucene_path = lucene_docs_dir / "products.jsonl"

    seen_products: set[str] = set()
    seen_queries: dict[str, set[str]] = {"train": set(), "test": set()}
    stats: dict[str, Any] = {
        "raw_files": [path.name for path in parquet_paths],
        "rows": {"train": 0, "test": 0, "total": 0},
        "unique_queries": {"train": 0, "test": 0, "total": 0},
        "unique_products": 0,
        "outputs": {},
    }

    with products_path.open("w", encoding="utf-8") as products_f, lucene_path.open("w", encoding="utf-8") as lucene_f:
        qrels_handles = {split: path.open("w", encoding="utf-8") for split, path in qrels_paths.items()}
        queries_handles = {split: path.open("w", encoding="utf-8") for split, path in queries_paths.items()}
        try:
            for _, batch in iter_batches(parquet_paths, batch_size=batch_size):
                row_count = len(batch["query_id"])
                for i in range(row_count):
                    split = clean_text(batch.get("split", [None] * row_count)[i]).lower()
                    if split not in qrels_handles:
                        continue
                    query_id = str(batch["query_id"][i])
                    query = clean_text(batch["query"][i])
                    product_id = clean_text(batch["product_id"][i])
                    label = clean_text(batch["esci_label"][i])

                    stats["rows"][split] += 1
                    stats["rows"]["total"] += 1

                    if query_id not in seen_queries[split]:
                        seen_queries[split].add(query_id)
                        write_jsonl_record(queries_handles[split], {"query_id": query_id, "query": query, "split": split})

                    write_jsonl_record(
                        qrels_handles[split],
                        {"query_id": query_id, "query": query, "product_id": product_id, "esci_label": label, "split": split},
                    )

                    if product_id and product_id not in seen_products:
                        raw_product = {col: batch.get(col, [None] * row_count)[i] for col in PRODUCT_COLUMNS}
                        product_record = {
                            "product_id": product_id,
                            "product_locale": clean_text(raw_product.get("product_locale")),
                            "product_title": clean_text(raw_product.get("product_title")),
                            "product_description": clean_text(raw_product.get("product_description")),
                            "product_bullet_point": clean_text(raw_product.get("product_bullet_point")),
                            "product_brand": clean_text(raw_product.get("product_brand")),
                            "product_color": clean_text(raw_product.get("product_color")),
                        }
                        product_record["product_text"] = build_product_text(product_record)
                        seen_products.add(product_id)
                        write_jsonl_record(products_f, product_record)
                        write_jsonl_record(lucene_f, {"id": product_id, "contents": product_record["product_text"], **product_record})
        finally:
            for handle in qrels_handles.values():
                handle.close()
            for handle in queries_handles.values():
                handle.close()

    total_queries = set().union(*seen_queries.values())
    stats["unique_queries"] = {"train": len(seen_queries["train"]), "test": len(seen_queries["test"]), "total": len(total_queries)}
    stats["unique_products"] = len(seen_products)
    stats["outputs"] = {
        "products": str(products_path),
        "lucene_docs": str(lucene_path),
        "queries_train": str(queries_paths["train"]),
        "queries_test": str(queries_paths["test"]),
        "qrels_train": str(qrels_paths["train"]),
        "qrels_test": str(qrels_paths["test"]),
    }
    (output_dir / "prepare_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ESCI English small data for Phase 1 indexing.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=25_000)
    args = parser.parse_args()
    stats = prepare_esci(args.raw_dir, args.output_dir, args.batch_size)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()