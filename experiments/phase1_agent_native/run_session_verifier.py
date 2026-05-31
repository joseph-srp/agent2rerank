from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.agent.backends import OpenAICompatibleBackend
from src.agent.schemas import VerifierResult
from src.agent.tools import DocumentTool, iter_jsonl
from src.utils.config import load_yaml
from src.verifier.session_verifier import SessionVerifier, write_verifier_results


class MockVerifierBackend:
    def chat(self, messages, tools=None):
        payload = json.loads(messages[-1]["content"])
        trajectory = payload["trajectory"]
        success = bool(trajectory.get("selected_product_id")) and bool(trajectory.get("rejected_products")) and not trajectory.get("parse_error")
        return {"role": "assistant", "content": json.dumps({"verifier_success": success, "verifier_score": 4 if success else 1, "verifier_confidence": 0.9 if success else 0.2, "rationale": "mock structural verifier"})}


def default_input(config: dict, split: str) -> Path:
    name = "phase1_dev_trajectories.jsonl" if split in {"dev", "test"} else "phase1_train_trajectories.jsonl"
    return Path(config["data"]["outputs_dir"]) / "trajectories" / name


def default_output(config: dict, split: str) -> Path:
    name = "phase1_dev_verifier.jsonl" if split in {"dev", "test"} else "phase1_train_verifier.jsonl"
    return Path(config["data"]["outputs_dir"]) / "verifier_results" / name


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Phase 1 shopping-agent sessions.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase1_agent_lrat_style.yaml")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="train")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mock", action="store_true", help="Use structural mock verifier instead of an LLM.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    backend = MockVerifierBackend() if args.mock else OpenAICompatibleBackend.from_config(config["llm"])
    verifier = SessionVerifier(backend, DocumentTool(config["data"]["products_path"]))
    stats = write_verifier_results(args.input or default_input(config, args.split), args.output or default_output(config, args.split), verifier)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
