from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agent.tools import DocumentTool, iter_jsonl


def verifier_is_success(record: dict[str, Any], min_score: int = 3, min_confidence: float = 0.7) -> bool:
    return bool(record.get("verifier_success")) and int(record.get("verifier_score", 0)) >= min_score and float(record.get("verifier_confidence", 0.0)) >= min_confidence


class PairBuilder:
    def __init__(self, document_tool: DocumentTool, *, min_score: int = 3, min_confidence: float = 0.7):
        self.document_tool = document_tool
        self.min_score = min_score
        self.min_confidence = min_confidence

    def build_pairs(self, trajectory: dict[str, Any], verifier_result: dict[str, Any]) -> list[dict[str, Any]]:
        if not verifier_is_success(verifier_result, self.min_score, self.min_confidence):
            return []
        query_id = str(trajectory.get("query_id", ""))
        selected_id = str(trajectory.get("selected_product_id", ""))
        if not query_id or not selected_id:
            return []
        try:
            positive = self.document_tool.get_document(selected_id)
        except KeyError:
            return []

        pairs: list[dict[str, Any]] = []
        for rejected in trajectory.get("rejected_products", []):
            negative_id = str(rejected.get("product_id", ""))
            if not negative_id or negative_id == selected_id:
                continue
            try:
                negative = self.document_tool.get_document(negative_id)
            except KeyError:
                continue
            pairs.append(
                {
                    "pair_id": f"{query_id}_{selected_id}_gt_{negative_id}",
                    "query_id": query_id,
                    "original_query": trajectory.get("original_query", ""),
                    "agent_search_query": trajectory.get("agent_search_query", trajectory.get("original_query", "")),
                    "query_for_training": trajectory.get("agent_search_query", trajectory.get("original_query", "")),
                    "positive_product_id": selected_id,
                    "positive_text": positive.get("product_text", ""),
                    "negative_product_id": negative_id,
                    "negative_text": negative.get("product_text", ""),
                    "agent_rejection_reason": rejected.get("reason", ""),
                    "source": "successful_session_selected_vs_rejected",
                    "verifier_score": verifier_result.get("verifier_score"),
                    "verifier_confidence": verifier_result.get("verifier_confidence"),
                }
            )
        return pairs


def build_pairs_file(
    trajectories_path: str | Path,
    verifier_path: str | Path,
    output_path: str | Path,
    builder: PairBuilder,
) -> dict[str, Any]:
    trajectories = {record["query_id"]: record for record in iter_jsonl(trajectories_path)}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sessions = 0
    pairs = 0
    with output.open("w", encoding="utf-8") as out:
        for verifier_result in iter_jsonl(verifier_path):
            query_id = verifier_result.get("query_id")
            trajectory = trajectories.get(query_id)
            if not trajectory:
                continue
            sessions += 1
            for pair in builder.build_pairs(trajectory, verifier_result):
                pairs += 1
                out.write(json.dumps(pair, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"trajectories": str(trajectories_path), "verifier": str(verifier_path), "output": str(output), "sessions": sessions, "pairs": pairs}
