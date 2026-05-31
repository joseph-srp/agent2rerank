# LRAT-Style Shopping Agent

Phase 1 uses a lightweight LRAT-style ReAct/tool-calling loop instead of a LangGraph-first implementation. The stable interface is the trajectory schema, so future framework adapters can preserve downstream verifier and pair-builder behavior.

## Tools

The agent has two tools:

```text
search(query, top_k?)
get_document(product_id)
```

`search` calls the existing Pyserini/Lucene BM25 index and returns ranked snippets with `rank`, `product_id`, `bm25_score`, and `snippet`. `get_document` returns full ESCI product metadata from `products.jsonl`.

The final shopping decision is not a tool call. The model must emit valid JSON with `query_id`, `original_query`, `agent_search_query`, `opened_products`, `rejected_products`, `selected_product_id`, and `final_answer`. Internal tool calls are recorded in `tool_calls`.

## Outputs

Trajectory scripts write JSONL to:

```text
/root/autodl-tmp/agent2rerank-data/outputs/trajectories/
```

Verifier scripts write JSONL to:

```text
/root/autodl-tmp/agent2rerank-data/outputs/verifier_results/
```

Pair construction writes JSONL to:

```text
/root/autodl-tmp/agent2rerank-data/outputs/pairs/
```
