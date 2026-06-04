# Transition Notes

This project keeps code in Git, but the ESCI data, Lucene index, run outputs, Python environments, and model weights live outside the repository under `/root/autodl-tmp/agent2rerank-data/`. These artifacts should be copied to a new machine or regenerated with the commands below.

## External Artifacts

Current external root:

```text
/root/autodl-tmp/agent2rerank-data/
```

Important subdirectories:

```text
raw/esci_english_small/                         # downloaded ESCI parquet files
processed/esci_english_small/                   # JSONL products, queries, qrels, Lucene docs
indexes/esci_english_small_pyserini_lucene/     # Pyserini/Lucene BM25 index
outputs/bm25_runs/                              # BM25 top100 train/test runs
outputs/rerank_runs/                            # Qwen3 rerank runs and smoke outputs
outputs/eval/                                   # reranker metric JSON files
outputs/trajectories/                           # agent trajectory JSONL outputs
outputs/verifier_results/                       # verifier JSONL outputs
outputs/pairs/                                  # selected-vs-rejected training pairs
models/                                         # Hugging Face model cache for Qwen3 reranker
envs/                                           # local venvs; usually rebuild instead of copying
```

Do not commit these artifacts. `.gitignore` keeps repo-local `data/` and `outputs/` as placeholders only.

## Downloaded Data Files

The raw ESCI English small files currently present are:

```text
/root/autodl-tmp/agent2rerank-data/raw/esci_english_small/README.md
/root/autodl-tmp/agent2rerank-data/raw/esci_english_small/SHA256SUMS
/root/autodl-tmp/agent2rerank-data/raw/esci_english_small/dataset_stats.json
/root/autodl-tmp/agent2rerank-data/raw/esci_english_small/train-00000-of-00002.parquet
/root/autodl-tmp/agent2rerank-data/raw/esci_english_small/train-00001-of-00002.parquet
/root/autodl-tmp/agent2rerank-data/raw/esci_english_small/test-00000-of-00001.parquet
```

Approximate parquet sizes:

```text
train-00000-of-00002.parquet  134 MB
train-00001-of-00002.parquet  131 MB
test-00000-of-00001.parquet   116 MB
```

If these files are not copied to the new machine, download the same ESCI English small parquet files again into the raw directory above before running data preparation.

## Data Preparation Pipeline

The raw parquet files are converted by:

```bash
cd /root/agent2rerank
python experiments/phase1_agent_native/prepare_esci.py \
  --raw-dir /root/autodl-tmp/agent2rerank-data/raw/esci_english_small \
  --output-dir /root/autodl-tmp/agent2rerank-data/processed/esci_english_small
```

This calls `src/data/esci_prepare.py`. It reads the parquet columns for query/product/qrel data, strips HTML, normalizes whitespace, builds `product_text` as:

```text
Title: ... Brand: ... Color: ... Bullet points: ... Description: ...
```

Generated processed files:

```text
products.jsonl
queries_train.jsonl
queries_test.jsonl
qrels_train.jsonl
qrels_test.jsonl
lucene_docs/products.jsonl
prepare_stats.json
```

Current expected counts:

```text
products.jsonl        482,105 products
queries_train.jsonl    20,888 queries
queries_test.jsonl      8,956 queries
qrels_train.jsonl     419,653 judgments
qrels_test.jsonl      181,701 judgments
```

## BM25 Index And Runs

Build the Lucene index from processed Lucene docs:

```bash
cd /root/agent2rerank
python experiments/phase1_agent_native/build_bm25_index.py \
  --input-dir /root/autodl-tmp/agent2rerank-data/processed/esci_english_small/lucene_docs \
  --index-dir /root/autodl-tmp/agent2rerank-data/indexes/esci_english_small_pyserini_lucene \
  --threads 4
```

The index uses Pyserini/Lucene with stored raw documents. Current index size is about 620 MB.

Generate BM25 top100 runs:

```bash
cd /root/agent2rerank
python experiments/phase1_agent_native/run_bm25.py \
  --index-dir /root/autodl-tmp/agent2rerank-data/indexes/esci_english_small_pyserini_lucene \
  --queries /root/autodl-tmp/agent2rerank-data/processed/esci_english_small/queries_train.jsonl \
  --output /root/autodl-tmp/agent2rerank-data/outputs/bm25_runs/esci_train_bm25_top100.jsonl \
  --top-k 100 --k1 0.9 --b 0.4

python experiments/phase1_agent_native/run_bm25.py \
  --index-dir /root/autodl-tmp/agent2rerank-data/indexes/esci_english_small_pyserini_lucene \
  --queries /root/autodl-tmp/agent2rerank-data/processed/esci_english_small/queries_test.jsonl \
  --output /root/autodl-tmp/agent2rerank-data/outputs/bm25_runs/esci_test_bm25_top100.jsonl \
  --top-k 100 --k1 0.9 --b 0.4
```

Current BM25 run sizes are roughly:

```text
esci_train_bm25_top100.jsonl  3.6 GB
esci_test_bm25_top100.jsonl   1.6 GB
```

## Qwen3 Reranker Weights

The reranker uses:

```text
Qwen/Qwen3-Reranker-0.6B
```

Configured cache path:

```text
/root/autodl-tmp/agent2rerank-data/models
```

The weights were downloaded through Hugging Face during real reranker smoke tests. Current cache size is about 1.2 GB. If the cache is not migrated, it will be downloaded again when running `run_qwen3_rerank.py`.

If the server needs a local proxy, export it before running Qwen3 commands:

```bash
export http_proxy=http://127.0.0.1:17890
export https_proxy=http://127.0.0.1:17890
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
```

Qwen3 rerank smoke command:

```bash
cd /root/agent2rerank
python experiments/phase1_agent_native/run_qwen3_rerank.py \
  --split test \
  --limit 3 \
  --top-k 10 \
  --batch-size 2 \
  --output /root/autodl-tmp/agent2rerank-data/outputs/rerank_runs/qwen3_real_smoke_3q_top10.jsonl
```

Evaluate a rerank run:

```bash
python experiments/phase1_agent_native/evaluate_rerank.py \
  --split test \
  --run /root/autodl-tmp/agent2rerank-data/outputs/rerank_runs/qwen3_real_smoke_3q_top10.jsonl \
  --output /root/autodl-tmp/agent2rerank-data/outputs/eval/qwen3_real_smoke_3q_top10_metrics.json
```

## Python Environments

Prefer rebuilding environments on a new server instead of copying `envs/`.

CPU BM25/Pyserini environment:

```bash
cd /root/agent2rerank
python3 -m venv /root/autodl-tmp/agent2rerank-data/envs/pyserini
source /root/autodl-tmp/agent2rerank-data/envs/pyserini/bin/activate
pip install -r requirements.txt
```

GPU Qwen3 reranker environment:

```bash
cd /root/agent2rerank
python3 -m venv /root/autodl-tmp/agent2rerank-data/envs/reranker-gpu
source /root/autodl-tmp/agent2rerank-data/envs/reranker-gpu/bin/activate
mkdir -p /root/autodl-tmp/pip-cache /root/autodl-tmp/pip-tmp
TMPDIR=/root/autodl-tmp/pip-tmp PIP_CACHE_DIR=/root/autodl-tmp/pip-cache pip install -r requirements-reranker-gpu.txt
```

The GPU environment was tested with Python 3.10, torch 2.5.1+cu124, Transformers 4.57.6, and an RTX 4090 D. Make sure `torch.cuda.is_available()` is true before running full reranking.

## Minimal New-Machine Checklist

1. Clone repo and checkout `phase1`.
2. Rebuild CPU and GPU Python environments from the two requirements files.
3. Copy or re-download raw ESCI parquet files.
4. Run `prepare_esci.py` if processed JSONL files are missing.
5. Build BM25 index if the Lucene index is missing.
6. Run `run_bm25.py` if BM25 top100 files are missing.
7. Copy or re-download Qwen3 weights into `/root/autodl-tmp/agent2rerank-data/models`.
8. Run a Qwen3 rerank smoke test and evaluation.
9. For agent testing with reranked candidates, pass `--reranked-run` to `run_agent_trajectory.py`.

Example agent smoke with a reranked run:

```bash
python experiments/phase1_agent_native/run_agent_trajectory.py \
  --split test \
  --limit 1 \
  --mock \
  --reranked-run /root/autodl-tmp/agent2rerank-data/outputs/rerank_runs/qwen3_real_smoke_3q_top10.jsonl \
  --output /root/autodl-tmp/agent2rerank-data/outputs/trajectories/qwen3_reranked_agent_mock_smoke.jsonl
```
