# Amazon ESCI English Small

## Local Data

Raw ESCI English small parquet files are stored outside the Git repository on the data disk:

```text
/root/autodl-tmp/agent2rerank-data/raw/esci_english_small/
```

Downloaded files:

```text
test-00000-of-00001.parquet   116M
train-00000-of-00002.parquet  134M
train-00001-of-00002.parquet  131M
```

The raw directory also contains `SHA256SUMS`, `README.md`, and `dataset_stats.json`.

## Dataset Statistics

English small total:

```text
rows / judgements: 601,354
unique queries:    29,844
unique products:   482,105
locale:            us
```

Train split:

```text
rows / judgements: 419,653
unique queries:    20,888
unique products:   351,961
```

Test split:

```text
rows / judgements: 181,701
unique queries:    8,956
unique products:   164,900
```

Label distribution:

```text
E: 261,527
S: 211,191
I: 101,447
C: 27,189
```

Judgements per query:

```text
min:    8
mean:   20.15
median: 16
max:    188
```

## Columns

Useful columns for Phase 1:

```text
example_id
query
query_id
product_id
product_locale
esci_label
small_version
large_version
split
product_title
product_description
product_bullet_point
product_brand
product_color
```

`esci_label` values:

```text
E = Exact
S = Substitute
C = Complement
I = Irrelevant
```

## Phase 1 Usage Rules

ESCI labels are evaluation-only in Phase 1. They must not be used for BM25 indexing, agent trajectory construction, session filtering, or selected-versus-rejected pair construction.

The searchable product catalog is built from all unique products appearing in the English small train and test parquet files. Each unique `product_id` becomes one Lucene document.

## Product Text

BM25 and the Qwen3 reranker should share the same `product_text` representation:

```text
Title: ... Brand: ... Color: ... Bullet points: ... Description: ...
```

HTML tags are stripped, HTML entities are unescaped, and whitespace is normalized.

## Generated Phase 1 Files

Processed files are written to:

```text
/root/autodl-tmp/agent2rerank-data/processed/esci_english_small/
```

Generated processed files:

```text
products.jsonl                    482,105 lines
queries_train.jsonl                20,888 lines
queries_test.jsonl                  8,956 lines
qrels_train.jsonl                 419,653 lines
qrels_test.jsonl                  181,701 lines
lucene_docs/products.jsonl        482,105 lines
prepare_stats.json
```

The Pyserini/Lucene BM25 index is written to:

```text
/root/autodl-tmp/agent2rerank-data/indexes/esci_english_small_pyserini_lucene/
```

Index build result:

```text
indexed documents: 482,105
unindexable:       0
empty:             0
errors:            0
index size:        about 620M
```

BM25 retrieval outputs are written to:

```text
/root/autodl-tmp/agent2rerank-data/outputs/bm25_runs/
```

Generated BM25 runs:

```text
esci_train_bm25_top100.jsonl   20,888 lines   about 3.6G
esci_test_bm25_top100.jsonl     8,956 lines   about 1.6G
```

Each BM25 record contains `query_id`, `query`, and `bm25_top100`. Each hit contains `rank`, `product_id`, `bm25_score`, and `product_text`.

## BM25 Environment

BM25 retrieval runs on CPU. It does not use GPU acceleration. The current retrieval stack is:

```text
OpenJDK 21
Python 3.10 venv: /root/autodl-tmp/agent2rerank-data/envs/pyserini
Pyserini 0.37.0
Lucene default English analyzer
BM25 k1=0.9, b=0.4
```

Qwen3 reranking should use a separate GPU-capable environment later, so the BM25 environment remains stable and CPU-only.
## Phase 1 Agent Outputs

Agent trajectories, verifier outputs, and pair data are written outside Git under `/root/autodl-tmp/agent2rerank-data/outputs/` in `trajectories/`, `verifier_results/`, and `pairs/`. ESCI labels remain evaluation-only for these stages.
