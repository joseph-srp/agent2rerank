from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    step: int
    tool: str
    arguments: dict[str, Any]
    result_summary: dict[str, Any]


@dataclass
class RejectedProduct:
    product_id: str
    reason: str


@dataclass
class Trajectory:
    query_id: str
    original_query: str
    agent_search_query: str
    opened_products: list[str]
    rejected_products: list[RejectedProduct]
    selected_product_id: str
    final_answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    raw_final_response: str | None = None
    parse_error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryRecord:
    query_id: str
    query: str
    split: str | None = None


@dataclass
class VerifierResult:
    query_id: str
    selected_product_id: str
    verifier_success: bool
    verifier_score: int
    verifier_confidence: float
    rationale: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
