from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agent.backends import AgentBackend
from src.agent.output_parser import extract_json_object
from src.agent.schemas import VerifierResult
from src.agent.tools import DocumentTool, iter_jsonl
from src.verifier.verifier_prompts import SESSION_VERIFIER_PROMPT


class SessionVerifier:
    def __init__(self, backend: AgentBackend, document_tool: DocumentTool):
        self.backend = backend
        self.document_tool = document_tool

    def verify(self, trajectory: dict[str, Any]) -> VerifierResult:
        selected_id = trajectory.get("selected_product_id", "")
        product_ids = [selected_id] + [item.get("product_id", "") for item in trajectory.get("rejected_products", [])]
        products: dict[str, Any] = {}
        for product_id in dict.fromkeys(pid for pid in product_ids if pid):
            try:
                products[product_id] = self.document_tool.get_document(product_id)
            except KeyError:
                products[product_id] = {"product_id": product_id, "error": "not found"}
        messages = [
            {"role": "system", "content": SESSION_VERIFIER_PROMPT},
            {
                "role": "user",
                "content": json.dumps({"trajectory": trajectory, "products": products}, ensure_ascii=False),
            },
        ]
        message = self.backend.chat(messages, tools=None)
        data = extract_json_object(message.get("content") or "{}")
        return VerifierResult(
            query_id=str(trajectory.get("query_id", "")),
            selected_product_id=str(selected_id),
            verifier_success=bool(data.get("verifier_success", False)),
            verifier_score=int(data.get("verifier_score", 0)),
            verifier_confidence=float(data.get("verifier_confidence", 0.0)),
            rationale=str(data.get("rationale", "")),
        )


def write_verifier_results(trajectories_path: str | Path, output_path: str | Path, verifier: SessionVerifier) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    success = 0
    with output.open("w", encoding="utf-8") as out:
        for trajectory in iter_jsonl(trajectories_path):
            result = verifier.verify(trajectory)
            if result.verifier_success:
                success += 1
            count += 1
            out.write(json.dumps(result.to_json(), ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"input": str(trajectories_path), "output": str(output), "sessions": count, "verifier_success": success}
