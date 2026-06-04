from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.reranking.qwen3_reranker import BM25ScoreMockReranker, Qwen3HitScorer, Qwen3Reranker, Qwen3RerankerConfig
from src.reranking.run_rerank import run_rerank_file
from src.utils.config import load_yaml


def path_for_split(config: dict, split: str, key: str) -> Path:
    split_config = config["data"][split]
    return Path(split_config[key])


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank BM25 runs with Qwen3-Reranker-0.6B.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase1_qwen3_reranker.yaml")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None, help="Number of BM25 candidates to rerank per query.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="Use BM25 scores as deterministic mock reranker scores.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    reranker_config = config["reranker"]
    batch_size = args.batch_size or int(reranker_config.get("batch_size", 8))
    candidate_top_k = args.top_k or int(reranker_config.get("candidate_top_k", 100))
    scorer = BM25ScoreMockReranker()
    if not args.mock:
        scorer = Qwen3HitScorer(
            Qwen3Reranker(
                Qwen3RerankerConfig(
                    model_name_or_path=reranker_config.get("model_name_or_path", "Qwen/Qwen3-Reranker-0.6B"),
                    device=reranker_config.get("device", "auto"),
                    dtype=reranker_config.get("dtype", "auto"),
                    cache_dir=reranker_config.get("cache_dir"),
                    max_length=int(reranker_config.get("max_length", 8192)),
                    instruction=reranker_config.get("instruction"),
                )
            )
        )
    stats = run_rerank_file(
        args.input or path_for_split(config, args.split, "bm25_run_path"),
        args.output or path_for_split(config, args.split, "rerank_output_path"),
        scorer,
        candidate_top_k=candidate_top_k,
        batch_size=batch_size,
        limit=args.limit,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
