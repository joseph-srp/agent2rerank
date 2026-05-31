from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.agent.tools import DocumentTool
from src.signal.pair_builder import PairBuilder, build_pairs_file
from src.utils.config import load_yaml


def path_for(config: dict, kind: str, split: str) -> Path:
    suffix = "dev" if split in {"dev", "test"} else "train"
    filename = {
        "trajectories": f"phase1_{suffix}_trajectories.jsonl",
        "verifier_results": f"phase1_{suffix}_verifier.jsonl",
        "pairs": f"phase1_agent_pairs_{suffix}.jsonl",
    }[kind]
    return Path(config["data"]["outputs_dir"]) / kind / filename


def main() -> None:
    parser = argparse.ArgumentParser(description="Build selected-vs-rejected reranker pairs from successful sessions.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase1_agent_lrat_style.yaml")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="train")
    parser.add_argument("--trajectories", type=Path, default=None)
    parser.add_argument("--verifier", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    builder = PairBuilder(DocumentTool(config["data"]["products_path"]))
    stats = build_pairs_file(
        args.trajectories or path_for(config, "trajectories", args.split),
        args.verifier or path_for(config, "verifier_results", args.split),
        args.output or path_for(config, "pairs", args.split),
        builder,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
