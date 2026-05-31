from __future__ import annotations

import json
from typing import Any

from src.agent.backends import AgentBackend
from src.agent.output_parser import TrajectoryParseError, parse_final_trajectory
from src.agent.schemas import QueryRecord, ToolCallRecord, Trajectory
from src.agent.tools import DocumentTool, SearchTool, openai_tool_specs


SYSTEM_PROMPT = """You are a shopping search agent. Use the tools to search products and open product documents.
Decide which product best satisfies the user's original shopping query. Open products before selecting or rejecting them.
When done, output only valid JSON with this schema:
{
  "query_id": "...",
  "original_query": "...",
  "agent_search_query": "...",
  "opened_products": ["..."],
  "rejected_products": [{"product_id": "...", "reason": "..."}],
  "selected_product_id": "...",
  "final_answer": "..."
}
Do not call a tool after producing the final JSON."""


class LRATStyleAgent:
    def __init__(
        self,
        backend: AgentBackend,
        search_tool: SearchTool,
        document_tool: DocumentTool,
        *,
        max_turns: int = 8,
        max_opened_products: int = 5,
    ):
        self.backend = backend
        self.search_tool = search_tool
        self.document_tool = document_tool
        self.max_turns = max_turns
        self.max_opened_products = max_opened_products

    def run(self, query: QueryRecord) -> Trajectory:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"query_id: {query.query_id}\noriginal_query: {query.query}\nFind the best matching product.",
            },
        ]
        tool_calls: list[ToolCallRecord] = []
        opened_products: list[str] = []
        final_text = ""

        for _turn in range(1, self.max_turns + 1):
            message = self.backend.chat(messages, tools=openai_tool_specs())
            messages.append(message)
            calls = message.get("tool_calls") or []
            if not calls:
                final_text = message.get("content") or ""
                break

            for call in calls:
                function = call.get("function", {})
                tool_name = function.get("name")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                result = self._run_tool(tool_name, arguments, opened_products)
                summary = self._summarize_tool_result(tool_name, result)
                tool_calls.append(ToolCallRecord(step=len(tool_calls) + 1, tool=tool_name or "", arguments=arguments, result_summary=summary))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": tool_name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            if len(opened_products) >= self.max_opened_products:
                messages.append(
                    {
                        "role": "user",
                        "content": "You have opened the maximum number of products. Produce the final JSON now.",
                    }
                )

        if not final_text:
            final_message = self.backend.chat(messages, tools=None)
            final_text = final_message.get("content") or ""

        try:
            trajectory = parse_final_trajectory(final_text, query_id=query.query_id, original_query=query.query, tool_calls=tool_calls)
        except TrajectoryParseError as exc:
            trajectory = Trajectory(
                query_id=query.query_id,
                original_query=query.query,
                agent_search_query=query.query,
                opened_products=opened_products,
                rejected_products=[],
                selected_product_id="",
                final_answer="",
                tool_calls=tool_calls,
                raw_final_response=final_text,
                parse_error=str(exc),
            )
        return trajectory

    def _run_tool(self, tool_name: str | None, arguments: dict[str, Any], opened_products: list[str]) -> Any:
        if tool_name == "search":
            return self.search_tool.search(query=str(arguments.get("query", "")), top_k=arguments.get("top_k"))
        if tool_name == "get_document":
            product_id = str(arguments.get("product_id", ""))
            if len(opened_products) >= self.max_opened_products and product_id not in opened_products:
                return {"error": "max_opened_products reached", "product_id": product_id}
            try:
                result = self.document_tool.get_document(product_id)
            except KeyError as exc:
                return {"error": str(exc), "product_id": product_id}
            if product_id not in opened_products:
                opened_products.append(product_id)
            return result
        return {"error": f"unknown tool: {tool_name}"}

    @staticmethod
    def _summarize_tool_result(tool_name: str | None, result: Any) -> dict[str, Any]:
        if tool_name == "search" and isinstance(result, list):
            return {"num_results": len(result), "product_ids": [item.get("product_id") for item in result[:5]]}
        if tool_name == "get_document" and isinstance(result, dict):
            return {"product_id": result.get("product_id"), "error": result.get("error")}
        return {"type": type(result).__name__}
