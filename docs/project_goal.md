# Project Goal

## Overall Goal

The goal of this project is to study whether e-commerce search agents can help improve the underlying search system they rely on, especially the reranker.

Instead of treating the search agent only as a downstream user of the search system, this project explores the opposite direction: the agent’s own search behavior can become a source of training signal for the search system.

In a typical e-commerce search scenario, an agent issues search queries, inspects ranked products, opens product pages, rejects unsuitable candidates, and finally recommends or selects a product. These interactions naturally contain preference information. For example, if an agent selects one product after rejecting several others, the selected product can be treated as preferred over the rejected products under the same search context.

The central goal of this project is therefore:

> To develop a framework that converts e-commerce search agent trajectories into useful reranker training data, so that the lower-level search system can adapt and improve with reduced reliance on human-labeled or pair-level LLM-labeled relevance data.

## Motivation

E-commerce rerankers usually depend on query-product relevance supervision, such as human relevance labels, click logs, purchase logs, or LLM-generated labels. However, such supervision may be expensive, unavailable, or difficult to obtain in many vertical domains, such as industrial component search, scientific equipment search, enterprise procurement, medical device search, or small-scale specialized e-commerce platforms.

At the same time, search agents already generate rich interaction traces while solving shopping tasks. These traces include rewritten queries, opened products, rejected products, selected products, and rejection reasons. Such behavior can be viewed as a form of weak supervision.

This project investigates whether these agent-generated signals can be used to adapt an e-commerce reranker, allowing the search system and the search agent to improve together.

## Core Hypothesis

The core hypothesis is:

> Successful e-commerce search agent trajectories contain implicit pairwise preference signals that can be transformed into useful reranker training data.

In other words, if an agent successfully completes a search session, then the products it selects are likely to be better matches than the products it explicitly opens and rejects. These selected-versus-rejected product pairs can be used to fine-tune a reranker.

More broadly, this project assumes that agent behavior contains several types of useful signals:

* The agent’s rewritten or reformulated search queries reflect how the original user intent is operationalized for search.
* The products opened by the agent indicate candidates that appeared potentially relevant.
* The products rejected by the agent indicate hard negatives or near-miss candidates.
* The final selected product indicates the agent’s preferred result under the current search context.
* The agent’s rejection reasons can reveal why a candidate is unsuitable, such as category mismatch, attribute mismatch, price violation, compatibility mismatch, or complement-versus-target confusion.
* The final success or failure of a search session can provide coarse feedback about whether the trajectory should be trusted for training.

## Real-World Setting

In a real deployed e-commerce system, the success of an agent-generated search session can be estimated from user behavior, such as:

* Whether the user clicks the recommended product
* Whether the user stays on the product page
* Whether the user adds the product to cart
* Whether the user purchases the product
* Whether the user reformulates the query
* Whether the user abandons the session

These behavioral signals provide coarse session-level feedback. They do not require explicit query-product relevance annotation.

In offline experiments where real user behavior is unavailable, a verifier can be used to simulate this session-level feedback by judging whether the agent’s final recommendation satisfies the original search intent.

## Key Distinction

This project does not aim to prove that agent-generated supervision is always better than human-labeled or LLM-labeled supervision.

Instead, the goal is to show that:

> Agent trajectories can provide a practical and scalable source of weak supervision that reduces the need for pair-level relevance annotation.

The project shifts the supervision requirement from:

```text
pair-level query-product relevance labels
```

to:

```text
session-level feedback on whether a search was successful
```

This distinction is important because session-level feedback is often easier to obtain in real systems than dense query-product relevance labels.

The project may still compare against human-labeled or LLM-labeled supervision as reference settings, but the main value of the method is not necessarily to outperform them. The main value is to reduce the amount and granularity of external supervision required for reranker adaptation.

## High-Level Method Idea

The project studies a general loop:

```text
search agent uses search system
  ↓
search system returns ranked products
  ↓
agent searches, opens, rejects, and selects products
  ↓
session-level feedback identifies successful trajectories
  ↓
successful trajectories are converted into pairwise training data
  ↓
reranker is adapted using trajectory-derived supervision
  ↓
improved reranker supports future search agents
```

At a high level, the method is built around three ideas.

First, agent trajectories can be converted into pairwise preference data. A simple example is:

```text
selected product > opened-and-rejected product
```

Second, session-level feedback can be used to filter which trajectories are reliable enough for training. In a real system, this feedback may come from user behavior. In offline experiments, it may be simulated by a verifier.

Third, richer trajectory information can eventually be used to improve the quality of the generated supervision. For example, agent search queries can help define the query context of each pair, and rejection reasons can help identify different types of hard negatives.

## High-Level Extensions

Beyond the minimal trajectory-to-pair loop, the project may later explore several higher-level extensions.

### Effective Query Reconstruction

An agent’s actual search query may differ from the original user query. The agent may rewrite, shorten, or decompose the original request.

However, an agent-generated subquery may also omit important constraints from the original user intent. Therefore, a later extension is to reconstruct an effective query for each training pair by combining:

```text
original user query
+ agent search query
+ important constraints from the session
```

The goal is to attach each selected-versus-rejected product pair to the most appropriate search context.

### Failure Attribution

Not every failed search session should be used to train the reranker. A failure may be caused by different components of the system.

Possible failure sources include:

* Retrieval failure: the correct product was not retrieved.
* Reranking failure: the correct product was retrieved but ranked too low.
* Query understanding failure: the agent or search system used an incomplete or misleading query.
* Agent reasoning failure: the search results were adequate, but the agent chose or rejected products incorrectly.
* Metadata failure: product information was missing, incomplete, or misleading.

A later goal is to develop a failure attribution module that determines which failures are useful for reranker training and which should be assigned to other components, such as the retriever, query reformulator, agent reasoning module, or product metadata system.

### Typed Hard Negatives

Rejected products are not all the same. A product may be rejected because it is the wrong category, violates a budget constraint, lacks a required feature, is incompatible with the requested device, or is merely a complement rather than the target product.

A later extension is to use agent rejection reasons to classify rejected products into typed hard negatives, such as:

* Category mismatch
* Attribute mismatch
* Compatibility mismatch
* Price violation
* Brand mismatch
* Complement instead of target product
* Missing required feature

These typed hard negatives may help the reranker learn more fine-grained e-commerce relevance distinctions.

### Confidence and Feedback Weighting

Different trajectory signals may have different reliability. For example, a product explicitly opened and rejected by the agent may be a stronger negative than a product that was merely skipped. A session with strong positive user behavior may be more reliable than a session with weak or ambiguous feedback.

A later extension is to assign different weights to training pairs based on the strength of the trajectory evidence and session-level feedback.

### Agent-Search Co-Improvement

The long-term vision is to build an iterative improvement loop:

```text
better search system → better agent search results
better agent trajectories → better reranker training data
better reranker → better future search sessions
```

This turns the search agent from a passive consumer of search results into an active source of supervision for improving the search system.

## Expected Outcome

The expected outcome of this project is a working framework in which:

1. An e-commerce search agent interacts with a search system.
2. Successful search trajectories are identified using session-level feedback.
3. The agent’s selected and rejected products are converted into pairwise reranker training examples.
4. The reranker is fine-tuned using these trajectory-derived pairs.
5. The improved reranker helps future search agents retrieve and rank better products.

The long-term vision is to build an agent-search co-improvement loop:

```text
search system supports agent
agent behavior generates feedback
feedback improves reranker
improved reranker supports future agents
```

## Research Question

The main research question is:

> Can e-commerce search agents generate useful reranker training data from their own successful search trajectories, using only coarse session-level feedback instead of explicit pair-level relevance labels?

Additional high-level questions include:

* How much reranker improvement can be obtained from agent-generated trajectory supervision?
* Can session-level feedback replace dense pair-level relevance annotations for certain adaptation settings?
* Which parts of an agent trajectory are most useful for reranker training?
* How can failed trajectories be attributed to the correct component of the search-agent system?
* Can agent rejection reasons help construct better hard negatives?
* Can the search agent and the underlying search system improve each other over time?

A positive result would suggest that search agents can serve not only as users of search systems, but also as weak supervisors for improving them.
