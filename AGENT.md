# Agent Implementation Guide

This file gives coding agents the local project context and implementation boundaries for agent2rerank.

## Project Intent

Build a framework that converts e-commerce shopping agent trajectories into weak supervision for reranker adaptation. The main supervision signal is generated from successful agent sessions:

```text
selected product > opened-and-rejected product
```

The project is not trying to replace all human or LLM supervision. It is testing whether coarse session-level feedback plus agent behavior can reduce the need for dense pair-level relevance labels.

## Phase 1 Boundaries

Phase 1 should stay minimal and reproducible:

- Use ESCI as the product search dataset.
- Use BM25 for first-stage retrieval.
- Use Qwen3-Reranker-0.6B as the base reranker.
- Use an LRAT-style ReAct/tool-calling shopping agent first; preserve the trajectory schema so LangGraph or other adapters can be added later.
- Use a session-level verifier only to filter successful sessions.
- Build pairwise training data only from successful trajectories.
- Fine-tune the reranker with pairwise ranking loss, initially with LoRA.
- Evaluate both reranker-level metrics and agent-level verifier success.

Do not add later-phase ideas into Phase 1 unless explicitly requested.

## Important Constraints

For Phase 1 training pair construction, do not use:

- ESCI relevance labels.
- Human query-product relevance annotations.
- Pair-level LLM relevance labels.
- ESCI-label-based filtering.
- Effective query reconstruction.
- Failure attribution.
- Typed hard negative classification.
- Confidence-weighted loss.

ESCI labels are allowed for evaluation only.

## Expected Agent Behavior

The shopping search agent should be controlled and easy to audit. It should support these actions:

```text
search[query]
open[product_id]
reject[product_id, reason]
select[product_id]
stop
```

Phase 1 agent constraints:

- Inspect only top-k reranked products.
- Open at most 5 products per query.
- Select one product when possible.
- Provide reasons for rejected products.
- Emit valid structured JSON.

## Data Flow

The intended Phase 1 pipeline is:

```text
ESCI query
  -> BM25 retriever
  -> Qwen3-Reranker-0.6B reranked candidates
  -> LRAT-style shopping agent with search/get_document tools
  -> agent trajectory JSON
  -> session-level verifier JSON
  -> selected-vs-rejected pair JSONL
  -> pairwise reranker fine-tuning
  -> reranker and agent-level evaluation
```

## Directory Responsibilities

- `src/data/`: dataset loading, query splits, product text construction.
- `src/retrieval/`: BM25 indexing and retrieval.
- `src/reranking/`: Qwen scoring, pairwise training, ranking evaluation.
- `src/agent/`: LangGraph workflow, prompts, trajectory schema and logging.
- `src/verifier/`: session-level verifier prompts and parsing.
- `src/signal/`: successful trajectory filtering and pair construction.
- `src/utils/`: small shared utilities only.
- `experiments/phase1_agent_native/`: runnable scripts for the Phase 1 pipeline.
- `outputs/`: generated artifacts; do not commit large run outputs or checkpoints by default.

## Implementation Style

Keep code paths explicit and modular. Prefer small functions with clear input and output schemas. Use JSONL for large record-oriented artifacts. Validate structured LLM and agent outputs before writing them into downstream training data.

Avoid hidden coupling between stages. Each pipeline stage should read a documented input file and write a documented output file so intermediate artifacts can be inspected and regenerated.

## Evaluation Rules

Reranker-level evaluation may use ESCI labels to compute metrics such as NDCG@10, MRR@10, Hit@10 for Exact products, and Recall@10 for Exact products.

Agent-level evaluation should compare the same agent using the base reranker versus the agent-tuned reranker, judged by the same session-level verifier.

## Deferred Work

Leave these for later phases unless the user explicitly changes scope:

- Effective query reconstruction.
- Failure attribution.
- Typed hard negatives.
- Confidence-weighted pairwise loss.
- Pair-level LLM labeling baselines.
- Human-label-tuned baselines.
- WebShop or other interactive benchmarks.
- Multi-round self-improvement loops.

## Qwen3 Reranker Runtime

Qwen3 reranker inference is GPU-first and should use `requirements-reranker-gpu.txt`. The existing Pyserini environment remains CPU-oriented for indexing and BM25 retrieval.
