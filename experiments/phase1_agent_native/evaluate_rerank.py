from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.reranking.metrics import evaluate_rerank_file
from src.utils.config import load_yaml


def path_for_split(config: dict, split: str, key: str) -> Path:
    return Path(config["data"][split][key])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen3 rerank run with ESCI qrels.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase1_qwen3_reranker.yaml")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--qrels", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    config = load_yaml(args.config)
    result = evaluate_rerank_file(
        args.run or path_for_split(config, args.split, "rerank_output_path"),
        args.qrels or path_for_split(config, args.split, "qrels_path"),
        args.output or path_for_split(config, args.split, "metrics_output_path"),
        k=args.k,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
