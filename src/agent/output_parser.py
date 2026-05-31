from __future__ import annotations

import json
import re
from typing import Any

from src.agent.schemas import RejectedProduct, ToolCallRecord, Trajectory


class TrajectoryParseError(ValueError):
    pass


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise TrajectoryParseError("final response does not contain a JSON object")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise TrajectoryParseError(f"malformed final JSON: {exc}") from exc


def parse_final_trajectory(
    text: str,
    *,
    query_id: str,
    original_query: str,
    tool_calls: list[ToolCallRecord] | None = None,
) -> Trajectory:
    data = extract_json_object(text)
    required = ["selected_product_id", "opened_products", "rejected_products", "final_answer"]
    missing = [field for field in required if field not in data]
    if missing:
        raise TrajectoryParseError(f"missing required final JSON fields: {missing}")

    opened = data["opened_products"]
    if not isinstance(opened, list) or not all(isinstance(item, str) for item in opened):
        raise TrajectoryParseError("opened_products must be a list of product_id strings")

    rejected_raw = data["rejected_products"]
    if not isinstance(rejected_raw, list):
        raise TrajectoryParseError("rejected_products must be a list")
    rejected: list[RejectedProduct] = []
    for item in rejected_raw:
        if not isinstance(item, dict) or not isinstance(item.get("product_id"), str) or not isinstance(item.get("reason"), str):
            raise TrajectoryParseError("each rejected product must include string product_id and reason")
        rejected.append(RejectedProduct(product_id=item["product_id"], reason=item["reason"]))

    selected = data["selected_product_id"]
    if not isinstance(selected, str) or not selected:
        raise TrajectoryParseError("selected_product_id must be a non-empty string")

    return Trajectory(
        query_id=str(data.get("query_id") or query_id),
        original_query=str(data.get("original_query") or original_query),
        agent_search_query=str(data.get("agent_search_query") or original_query),
        opened_products=opened,
        rejected_products=rejected,
        selected_product_id=selected,
        final_answer=str(data["final_answer"]),
        tool_calls=tool_calls or [],
        raw_final_response=text,
    )
