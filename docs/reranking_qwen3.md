# Qwen3 Reranker Integration

Phase 1 reranking reads BM25 top-100 JSONL and writes a Qwen3 reranked JSONL run. ESCI labels are used only by the evaluation script.

Mock smoke test:

```bash
python experiments/phase1_agent_native/run_qwen3_rerank.py --split test --limit 3 --top-k 10 --mock
python experiments/phase1_agent_native/evaluate_rerank.py --split test --run /root/autodl-tmp/agent2rerank-data/outputs/rerank_runs/esci_test_qwen3_rerank_top100.jsonl
```

Real GPU smoke test:

```bash
python experiments/phase1_agent_native/run_qwen3_rerank.py --split test --limit 3 --top-k 10 --batch-size 2
```

Full output paths are configured in `configs/phase1_qwen3_reranker.yaml`.

`src.reranking.reranked_run.RerankedRunRetriever` can load a reranked JSONL run and expose a `search(query, top_k)` interface compatible with `SearchTool`.

## GPU Environment

Use `requirements-reranker-gpu.txt` in a GPU-capable environment. The tested setup is Python 3.10, torch 2.5.1+cu124, Transformers 4.57.6, and an RTX 4090 D. If the root filesystem is small, set `TMPDIR` and `PIP_CACHE_DIR` to `/root/autodl-tmp` during installation.

Hugging Face downloads can use the server-side proxy if available:

```bash
export http_proxy=http://127.0.0.1:17890
export https_proxy=http://127.0.0.1:17890
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
```
