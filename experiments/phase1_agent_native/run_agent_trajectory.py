from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.agent.backends import OpenAICompatibleBackend
from src.agent.lrat_style_agent import LRATStyleAgent
from src.agent.schemas import QueryRecord
from src.agent.tools import DocumentTool, SearchTool, iter_jsonl
from src.agent.trajectory_logger import TrajectoryLogger
from src.utils.config import load_yaml


class MockShoppingBackend:
    def __init__(self):
        self.turn = 0
        self.first_product = None
        self.second_product = None
        self.query_id = ""
        self.query = ""

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        self.turn += 1
        if self.turn == 1:
            content = messages[-1]["content"]
            self.query_id = content.split("query_id:", 1)[1].split("\n", 1)[0].strip()
            self.query = content.split("original_query:", 1)[1].split("\n", 1)[0].strip()
            return {"role": "assistant", "content": None, "tool_calls": [{"id": "call_search", "type": "function", "function": {"name": "search", "arguments": json.dumps({"query": self.query, "top_k": 2})}}]}
        if self.turn == 2:
            results = json.loads(messages[-1]["content"])
            self.first_product = results[0]["product_id"]
            self.second_product = results[1]["product_id"] if len(results) > 1 else results[0]["product_id"]
            return {"role": "assistant", "content": None, "tool_calls": [{"id": "call_doc1", "type": "function", "function": {"name": "get_document", "arguments": json.dumps({"product_id": self.first_product})}}, {"id": "call_doc2", "type": "function", "function": {"name": "get_document", "arguments": json.dumps({"product_id": self.second_product})}}]}
        content = {
            "query_id": self.query_id,
            "original_query": self.query,
            "agent_search_query": self.query,
            "opened_products": [self.first_product, self.second_product],
            "rejected_products": [{"product_id": self.second_product, "reason": "Less suitable than the selected product."}],
            "selected_product_id": self.first_product,
            "final_answer": "Selected the best available BM25 result after opening two products.",
        }
        return {"role": "assistant", "content": json.dumps(content)}


def query_path_for_split(config: dict[str, Any], split: str) -> Path:
    data = config["data"]
    if split == "train":
        return Path(data["queries_train_path"])
    if split == "dev":
        return Path(data.get("queries_dev_path") or data.get("queries_test_path") or data["queries_train_path"])
    if split == "test":
        return Path(data.get("queries_test_path") or data.get("queries_dev_path") or data["queries_train_path"])
    raise ValueError(f"unknown split: {split}")


def default_output(config: dict[str, Any], split: str) -> Path:
    name = "phase1_dev_trajectories.jsonl" if split in {"dev", "test"} else "phase1_train_trajectories.jsonl"
    return Path(config["data"]["outputs_dir"]) / "trajectories" / name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LRAT-style shopping trajectories over ESCI queries.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase1_agent_lrat_style.yaml")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mock", action="store_true", help="Use a deterministic mock backend for local smoke tests.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    retriever_config = config["retriever"]
    agent_config = config["agent"]
    search_tool = SearchTool.from_index(
        retriever_config["index_dir"],
        top_k=int(agent_config.get("search_top_k", 10)),
        k1=float(retriever_config.get("k1", 0.9)),
        b=float(retriever_config.get("b", 0.4)),
    )
    document_tool = DocumentTool(config["data"]["products_path"])
    backend = MockShoppingBackend() if args.mock else OpenAICompatibleBackend.from_config(config["llm"])
    agent = LRATStyleAgent(
        backend,
        search_tool,
        document_tool,
        max_turns=int(agent_config.get("max_turns", 8)),
        max_opened_products=int(agent_config.get("max_opened_products", 5)),
    )
    output = args.output or default_output(config, args.split)
    if output.exists():
        output.unlink()
    logger = TrajectoryLogger(output)

    count = 0
    parsed = 0
    for record in iter_jsonl(query_path_for_split(config, args.split)):
        trajectory = agent.run(QueryRecord(query_id=str(record["query_id"]), query=record["query"], split=record.get("split")))
        logger.write(trajectory.to_json())
        count += 1
        if not trajectory.parse_error:
            parsed += 1
        if args.limit and count >= args.limit:
            break
    print(json.dumps({"output": str(output), "sessions": count, "valid_final_json": parsed}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
