# agent2rerank

agent2rerank studies whether e-commerce search agent trajectories can be converted into useful weak supervision for adapting a product search reranker.

The central idea is that a shopping search agent does more than consume search results. During a session, it reformulates queries, opens products, rejects unsuitable candidates, and finally selects a recommendation. Successful sessions contain implicit preference signals: under the same search context, the selected product should be preferred over products the agent opened and rejected.

This project turns those trajectory signals into pairwise reranker training data, with the goal of reducing reliance on human-labeled or pair-level LLM-labeled query-product relevance annotations.

## Core Hypothesis

Successful e-commerce search agent trajectories contain implicit pairwise preference signals that can improve an underlying reranker.

A minimal training signal is:

```text
selected product > opened-and-rejected product
```

The project uses session-level feedback to decide which trajectories are reliable enough for training. In a real system, this feedback could come from clicks, dwell time, add-to-cart, purchase, reformulation, or abandonment. In offline experiments, an LLM verifier can simulate this coarse session-level feedback.

## Method Loop

```text
search agent uses search system
  -> search system returns ranked products
  -> agent searches, opens, rejects, and selects products
  -> session-level feedback identifies successful trajectories
  -> successful trajectories are converted into pairwise training data
  -> reranker is adapted using trajectory-derived supervision
  -> improved reranker supports future search agents
```

## Phase 1 Scope

Phase 1 focuses on building a minimal end-to-end system on the ESCI product search dataset:

1. Load ESCI queries and product metadata.
2. Retrieve candidates with BM25.
3. Rerank candidates with Qwen3-Reranker-0.6B.
4. Run a constrained LangGraph shopping search agent.
5. Record opened, rejected, and selected products.
6. Use a session-level verifier to keep successful sessions.
7. Build pairwise training examples from selected-versus-rejected products.
8. Fine-tune Qwen3-Reranker-0.6B with pairwise ranking loss.
9. Evaluate both reranker metrics and agent-level success.

ESCI labels are used for evaluation only. They should not be used to construct Phase 1 training pairs.

## Repository Structure

```text
configs/                    Experiment and model configuration files
src/data/                   ESCI loading and product text construction
src/retrieval/              BM25 retrieval
src/reranking/              Qwen reranking, training, and ranking evaluation
src/agent/                  LangGraph shopping agent and trajectory logging
src/verifier/               Session-level verifier and prompts
src/signal/                 Trajectory-to-pair training signal builders
src/utils/                  Shared IO, metrics, and logging helpers
experiments/phase1_agent_native/
                            Phase 1 runnable scripts
data/raw/                   Raw datasets
data/processed/             Processed ESCI products and query splits
outputs/bm25_runs/          BM25 retrieval outputs
outputs/rerank_runs/        Reranker outputs
outputs/trajectories/       Agent trajectory logs
outputs/verifier_results/   Session verifier outputs
outputs/pairs/              Pairwise training data
outputs/checkpoints/        Fine-tuned model checkpoints
outputs/eval/               Evaluation results
```

## Key Documents

- `docs/project_goal.md`: full project motivation, hypothesis, and long-term direction.
- `docs/phase1_plan.md`: concrete Phase 1 system design and implementation order.
- `AGENT.md`: implementation guidance for coding agents working in this repository.

## Success Criteria

Phase 1 is successful if the agent-tuned reranker improves over the base reranker on at least one main ranking metric and/or improves agent-level verifier success rate.

The strongest result is improvement on both ESCI ranking metrics and downstream agent recommendation success.
