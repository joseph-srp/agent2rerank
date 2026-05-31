# Phase 1 Plan: LRAT-Style Agent Trajectories for Reranker Adaptation

Phase 1 runs the complete closed loop:

```text
ESCI query
  -> BM25 retriever
  -> LRAT-style shopping agent
  -> structured trajectory
  -> session verifier
  -> selected-vs-rejected pair data
  -> Qwen3-Reranker-0.6B fine-tuning
  -> reranker-level + agent-level evaluation
```

The implemented Phase 1 agent is a ReAct/tool-calling loop with local product tools:

```text
search(query)
get_document(product_id)
```

The final shopping decision is structured JSON, not a tool. Each trajectory records the final answer plus the internal tool-call trace.

## Trajectory Schema

```json
{
  "query_id": "...",
  "original_query": "...",
  "agent_search_query": "...",
  "opened_products": ["..."],
  "rejected_products": [
    {"product_id": "...", "reason": "..."}
  ],
  "selected_product_id": "...",
  "final_answer": "...",
  "tool_calls": [
    {"step": 1, "tool": "search", "arguments": {"query": "..."}, "result_summary": {"num_results": 10}}
  ]
}
```

## Pair Construction

Only successful sessions are used. A session is successful when:

```text
verifier_success = true
verifier_score >= 3
verifier_confidence >= 0.7
```

Pairs are constructed as:

```text
positive = selected_product
negative = each opened_and_rejected_product
```

The Phase 1 default training query is `agent_search_query`; `original_query` is retained for later comparisons. ESCI labels are evaluation-only and are not used for trajectory generation, verifier filtering, or pair construction.

## Main Commands

Mock structural smoke test:

```bash
python experiments/phase1_agent_native/run_agent_trajectory.py --limit 3 --mock
python experiments/phase1_agent_native/run_session_verifier.py --mock
python experiments/phase1_agent_native/build_pairs.py
```

DeepSeek smoke test requires `DEEPSEEK_API_KEY` and an OpenAI-compatible endpoint in `configs/phase1_agent_lrat_style.yaml`:

```bash
python experiments/phase1_agent_native/run_agent_trajectory.py --limit 5
```

## Output Locations

```text
/root/autodl-tmp/agent2rerank-data/outputs/trajectories/phase1_train_trajectories.jsonl
/root/autodl-tmp/agent2rerank-data/outputs/trajectories/phase1_dev_trajectories.jsonl
/root/autodl-tmp/agent2rerank-data/outputs/verifier_results/phase1_train_verifier.jsonl
/root/autodl-tmp/agent2rerank-data/outputs/verifier_results/phase1_dev_verifier.jsonl
/root/autodl-tmp/agent2rerank-data/outputs/pairs/phase1_agent_pairs_train.jsonl
/root/autodl-tmp/agent2rerank-data/outputs/pairs/phase1_agent_pairs_dev.jsonl
```
