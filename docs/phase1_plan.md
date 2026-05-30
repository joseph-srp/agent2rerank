# Phase 1 Plan: Feedback-Filtered Agent Trajectory Supervision for E-commerce Reranker Adaptation

## 1. Phase 1 Goal

The goal of Phase 1 is to build a minimal working system to verify whether shopping search agent trajectories can provide useful weak supervision for adapting an e-commerce reranker.

In this phase, we do not aim to outperform human-labeled or LLM-labeled supervision. Instead, we aim to show that agent-generated trajectories, filtered by coarse session-level feedback, can produce training data that improves a reranker compared with the original base system.

The main research question is:

> Can a shopping search agent generate useful pairwise reranker training data from its own successful search trajectories, without requiring human-labeled or LLM-labeled query-product relevance annotations?

## 3. Phase 1 System Overview

The Phase 1 pipeline is:

```text
ESCI query
  ↓
BM25 retriever
  ↓
Qwen3-Reranker-0.6B
  ↓
Shopping search agent
  ↓
Agent trajectory:
  search query, opened products, rejected products, selected product
  ↓
LLM verifier simulates session-level user feedback
  ↓
Keep successful sessions only
  ↓
Construct pairwise training data:
  selected product > opened-and-rejected products
  ↓
Fine-tune Qwen3-Reranker-0.6B
  ↓
Evaluate reranker-level and agent-level performance
```

## 4. Dataset

### 4.1 Main Dataset

Use the ESCI product search dataset.

The dataset provides:

* User search queries
* Product metadata
* Query-product relevance labels: Exact, Substitute, Complement, Irrelevant
* Product text fields such as title, description, bullet points, brand, and color

### 4.2 How ESCI Labels Are Used

In Phase 1, ESCI labels are used only for evaluation, not for constructing training pairs.

Training pair construction should not use:

* ESCI relevance labels
* Human relevance annotations
* Pair-level LLM relevance labels

This keeps the main method aligned with the goal of avoiding pair-level external relevance supervision.

## 5. Search System

### 5.1 Retriever

Use BM25 as the retriever.

For each ESCI query:

```text
query → BM25 → top-100 candidate products
```

Recommended output file:

```text
outputs/bm25_runs/esci_bm25_top100.jsonl
```

Each record should contain:

```json
{
  "query_id": "...",
  "query": "...",
  "bm25_top100": [
    {
      "product_id": "...",
      "bm25_score": 0.0,
      "product_text": "..."
    }
  ]
}
```

### 5.2 Reranker

Use Qwen3-Reranker-0.6B as the reranker.

For each query, rerank the BM25 top-100 candidates:

```text
query + product_text → Qwen3-Reranker-0.6B score
```

Recommended output file:

```text
outputs/rerank_runs/base_qwen3_top100.jsonl
```

Each record should contain:

```json
{
  "query_id": "...",
  "query": "...",
  "reranked_top100": [
    {
      "product_id": "...",
      "rerank_score": 0.0,
      "rank": 1
    }
  ]
}
```

The shopping agent will only see the top-k products, for example top-10 or top-20.

## 6. Agent Design

### 6.1 Agent Framework

Use LangGraph to implement a controlled shopping search agent.

The goal is not to build a fully general autonomous shopping agent. Instead, the agent should follow a constrained workflow so that trajectories are easy to collect and convert into training data.

### 6.2 Agent Actions

The agent should support the following actions:

```text
search[query]
open[product_id]
reject[product_id, reason]
select[product_id]
stop
```

### 6.3 Agent Input

For each session, the agent receives:

* Original ESCI query
* Top-k reranked product summaries
* Product metadata for opened products

### 6.4 Agent Output

The agent should output structured JSON:

```json
{
  "query_id": "...",
  "original_query": "...",
  "agent_search_query": "...",
  "opened_products": ["p1", "p2", "p3"],
  "rejected_products": [
    {
      "product_id": "p1",
      "reason": "The product appears to be a complementary accessory rather than the target product."
    },
    {
      "product_id": "p2",
      "reason": "The product does not match the requested product type."
    }
  ],
  "selected_product_id": "p3",
  "final_answer": "I recommend product p3 because ..."
}
```

### 6.5 Agent Constraints

To keep Phase 1 simple and reproducible:

* The agent should inspect only top-k products.
* The agent should open at most 5 products per query.
* The agent must output one selected product when possible.
* The agent must provide reasons for rejected products.
* The agent output must be valid JSON.

## 7. Session-Level Verifier

### 7.1 Role of the Verifier

The verifier determines whether the agent’s final selected product satisfies the original query.

The verifier is used only to filter successful sessions. It does not label individual query-product pairs.

### 7.2 Real-World Interpretation

In a real deployed e-commerce search system, this verifier could be replaced by user behavior feedback such as:

* Clicks on the recommended product
* Dwell time
* Add-to-cart
* Purchase
* Bookmark or favorite
* Query reformulation
* Abandonment

In Phase 1, because ESCI is an offline benchmark without live user behavior, an LLM verifier is used as a proxy for session-level user feedback.

### 7.3 Verifier Input

The verifier receives:

```text
Original query
Selected product metadata
Agent final answer
```

### 7.4 Verifier Output

The verifier outputs JSON:

```json
{
  "success": true,
  "score": 4,
  "confidence": 0.86,
  "reason": "The selected product directly matches the original query."
}
```

### 7.5 Scoring Rubric

Use a 0–4 scale:

| Score | Meaning                                                                   |
| ----- | ------------------------------------------------------------------------- |
| 4     | Fully satisfies the query                                                 |
| 3     | Mostly satisfies the query, with minor uncertainty                        |
| 2     | Partially relevant but misses an important requirement                    |
| 1     | Related category or complementary product, but not a valid target product |
| 0     | Irrelevant or clearly wrong                                               |

Define session success as:

```text
success = score >= 3 and confidence >= 0.7
```

The threshold can be adjusted after preliminary inspection.

## 8. Training Pair Construction

### 8.1 Main Rule

For each successful session:

```text
selected product > opened-and-rejected product
```

That is:

```text
positive = agent selected product
negative = product opened and rejected by the agent
```

### 8.2 Training Pair Format

Each pair should be saved as JSONL:

```json
{
  "pair_id": "q001_p3_gt_p1",
  "query_id": "q001",

  "original_query": "iphone 13 case",
  "agent_search_query": "iphone 13 protective case",
  "query_for_training": "iphone 13 protective case",

  "positive_product_id": "p3",
  "positive_text": "Title: ... Brand: ... Description: ...",

  "negative_product_id": "p1",
  "negative_text": "Title: ... Brand: ... Description: ...",

  "agent_rejection_reason": "The rejected product is a screen protector rather than a phone case.",
  "source": "successful_session_selected_vs_rejected",

  "verifier_score": 4,
  "verifier_confidence": 0.86
}
```

### 8.3 What Not to Use in Phase 1

Phase 1 should not use:

* Effective query reconstruction
* Failure attribution
* Negative type classification
* Confidence-weighted loss
* ESCI-label-based pair filtering
* Pair-level LLM relevance labeling

These components can be explored in later phases.

## 9. Reranker Fine-Tuning

### 9.1 Model

Fine-tune Qwen3-Reranker-0.6B.

### 9.2 Training Objective

Use pairwise ranking loss:

```text
score(query, positive_product) > score(query, negative_product)
```

### 9.3 Training Query

Use the agent search query as the training query in Phase 1:

```text
query_for_training = agent_search_query
```

Also store the original query for future analysis.

Later phases can compare:

* Original query
* Agent search query
* Effective query

### 9.4 Training Strategy

Recommended initial setup:

```text
Training method: LoRA fine-tuning
Base model: Qwen3-Reranker-0.6B
Training data: successful trajectory pairs
Epochs: 1–3
Batch size: as hardware allows
Learning rate: start from 1e-5 or 2e-5
```

Start with a small subset to validate the pipeline before scaling.

## 10. Evaluation

Phase 1 should use two types of evaluation.

### 10.1 Reranker-Level Evaluation

Use ESCI labels only for evaluation.

For each test query:

```text
BM25 retrieve top-100
Qwen3 rerank top-100
Compute metrics using ESCI labels
```

Metrics:

```text
NDCG@10
MRR@10
Hit@10 for Exact products
Recall@10 for Exact products
```

Two relevance settings can be reported:

Binary relevance:

```text
Exact = 1
Others = 0
```

Graded relevance:

```text
Exact = 3
Substitute = 2
Complement = 1
Irrelevant = 0
```

### 10.2 Agent-Level Evaluation

Run the same shopping agent using:

1. Base BM25 + Qwen3-Reranker-0.6B
2. BM25 + Agent-Tuned Qwen3-Reranker-0.6B

Then use the same LLM verifier to judge final selected products.

Metrics:

```text
Verifier success rate
Average verifier score
Average number of opened products
Average number of rejected products
```

This evaluates whether the tuned reranker improves the downstream agent search experience.

## 11. Main Comparison

Phase 1 only needs a simple comparison:

| System      | Training Data                 | Pair-Level External Labels? | Session Verifier? | NDCG@10 | MRR@10 | Agent Success Rate |
| ----------- | ----------------------------- | --------------------------- | ----------------- | ------- | ------ | ------------------ |
| Base        | None                          | No                          | No                | -       | -      | -                  |
| Agent-Tuned | Successful agent trajectories | No                          | Yes               | -       | -      | -                  |

The expected outcome is:

```text
Agent-Tuned > Base
```

The goal is not to beat human-labeled or LLM-labeled supervision at this stage.

## 12. Optional Reference Baselines for Later

Not required in Phase 1, but useful for future phases:

| Method                | Supervision Source                          | Purpose                                       |
| --------------------- | ------------------------------------------- | --------------------------------------------- |
| Human-Label-Tuned     | ESCI labels                                 | Upper/reference supervised baseline           |
| LLM-Label-Tuned       | Pair-level LLM relevance labels             | External LLM supervision baseline             |
| Raw-Trajectory-Tuned  | All trajectories without verifier filtering | Test whether session feedback filtering helps |
| Effective-Query-Tuned | Agent query + recovered constraints         | Test effective query reconstruction           |
| Attribution-Tuned     | Failure-attribution-based pairs             | Test more precise pair construction           |

## 13. Code Organization

Recommended repository structure:

```text
agent2rank/
  README.md
  requirements.txt
  configs/
    phase1_agent_native.yaml

  src/
    data/
      esci_loader.py
      product_text_builder.py

    retrieval/
      bm25.py

    reranking/
      qwen3_reranker.py
      train_pairwise.py
      eval_ranking.py

    agent/
      langgraph_agent.py
      prompts.py
      trajectory_logger.py

    verifier/
      session_verifier.py
      verifier_prompts.py

    signal/
      pair_builder.py

    utils/
      io.py
      metrics.py
      logging.py

  experiments/
    phase1_agent_native/
      README.md
      run_bm25.py
      run_base_reranker.py
      run_agent_trajectory.py
      run_session_verifier.py
      build_pairs.py
      train_agent_tuned_reranker.py
      eval_reranker.py
      eval_agent_success.py

  data/
    raw/
    processed/

  outputs/
    bm25_runs/
    rerank_runs/
    trajectories/
    verifier_results/
    pairs/
    checkpoints/
    eval/
```

## 14. Suggested Implementation Order

### Step 1: Prepare ESCI

* Load ESCI data.
* Build product corpus.
* Construct product text.
* Split data by query ID into train/dev/test.

Output:

```text
data/processed/esci_products.jsonl
data/processed/esci_queries_train.jsonl
data/processed/esci_queries_dev.jsonl
data/processed/esci_queries_test.jsonl
```

### Step 2: Build BM25 Retriever

* Index product corpus.
* Retrieve top-100 products for each query.

Output:

```text
outputs/bm25_runs/esci_train_top100.jsonl
outputs/bm25_runs/esci_dev_top100.jsonl
outputs/bm25_runs/esci_test_top100.jsonl
```

### Step 3: Run Base Qwen3 Reranker

* Rerank BM25 top-100 candidates.
* Save reranked results.

Output:

```text
outputs/rerank_runs/base_qwen3_train_top100.jsonl
outputs/rerank_runs/base_qwen3_dev_top100.jsonl
outputs/rerank_runs/base_qwen3_test_top100.jsonl
```

### Step 4: Implement Shopping Agent

* Use LangGraph.
* Let the agent inspect top-k reranked products.
* Record opened, rejected, selected products.
* Save full trajectory.

Output:

```text
outputs/trajectories/phase1_train_trajectories.jsonl
```

### Step 5: Run Session-Level Verifier

* Verify whether the selected product satisfies the original query.
* Keep verifier output separate from trajectory logs.

Output:

```text
outputs/verifier_results/phase1_train_verifier.jsonl
```

### Step 6: Build Pairwise Training Data

* Keep only successful sessions.
* Construct selected > opened-and-rejected pairs.
* Do not use ESCI labels during pair construction.

Output:

```text
outputs/pairs/phase1_agent_pairs_train.jsonl
outputs/pairs/phase1_agent_pairs_dev.jsonl
```

### Step 7: Fine-Tune Qwen3-Reranker-0.6B

* Use pairwise ranking loss.
* Start with small-scale LoRA fine-tuning.
* Save checkpoint.

Output:

```text
outputs/checkpoints/qwen3_agent_tuned_phase1/
```

### Step 8: Evaluate Reranker

Compare:

```text
Base Qwen3-Reranker-0.6B
Agent-Tuned Qwen3-Reranker-0.6B
```

Use ESCI labels only for evaluation.

Output:

```text
outputs/eval/phase1_reranker_metrics.json
```

### Step 9: Evaluate Agent-Level Success

Run the same agent with:

```text
Base reranker
Agent-tuned reranker
```

Use the same LLM verifier to evaluate final recommendations.

Output:

```text
outputs/eval/phase1_agent_success_metrics.json
```

## 15. Phase 1 Deliverables

By the end of Phase 1, we should have:

1. A working BM25 + Qwen3-Reranker-0.6B search pipeline on ESCI.
2. A controlled LangGraph shopping search agent.
3. Agent trajectory logs.
4. Session-level verifier outputs.
5. Pairwise training data constructed from successful trajectories.
6. A fine-tuned Qwen3-Reranker-0.6B checkpoint.
7. Reranker-level evaluation results.
8. Agent-level success evaluation results.
9. A short analysis of whether agent trajectory supervision improves over the base reranker.

## 16. Phase 1 Success Criteria

Phase 1 is considered successful if:

```text
Agent-Tuned reranker improves over the Base reranker on at least one main ranking metric,
and/or improves agent-level verifier success rate.
```

The strongest expected result is:

```text
Agent-Tuned improves both ESCI ranking metrics and agent-level final recommendation success.
```

Even if the improvement is modest, Phase 1 is useful if it demonstrates that the full data-generation and training loop can run end to end.

## 17. What to Leave for Later Phases

The following components should be left for later:

* Effective query reconstruction
* Failure attribution
* Typed hard negatives
* Confidence-weighted pairwise loss
* Pair-level LLM labeling baseline
* Human-label-tuned baseline
* WebShop or other interactive benchmark
* More realistic user behavior simulation
* Multi-round self-improvement loop

Phase 1 should remain simple: build the full loop first, then refine the method.
