from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

DEFAULT_INSTRUCTION = "Given a user search query, determine whether the product document satisfies the query. Answer yes or no."


def format_qwen3_reranker_input(query: str, document: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def _dtype_from_name(torch_module, dtype: str):
    if dtype == "auto":
        return "auto"
    mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported dtype: {dtype}")
    return mapping[dtype]


@dataclass
class Qwen3RerankerConfig:
    model_name_or_path: str = "Qwen/Qwen3-Reranker-0.6B"
    device: str = "auto"
    dtype: str = "auto"
    cache_dir: str | None = None
    max_length: int = 8192
    instruction: str = DEFAULT_INSTRUCTION


class Qwen3Reranker:
    def __init__(self, config: Qwen3RerankerConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.torch = torch
        self.device = self._resolve_device(config.device)
        torch_dtype = _dtype_from_name(torch, config.dtype)
        model_kwargs = {"cache_dir": config.cache_dir} if config.cache_dir else {}
        if torch_dtype != "auto":
            model_kwargs["dtype"] = torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, padding_side="left", cache_dir=config.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.yes_token_id = self._token_id("yes")
        self.no_token_id = self._token_id("no")

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        return device

    def _token_id(self, token: str) -> int:
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
        ids = self.tokenizer.encode(" " + token, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
        raise ValueError(f"{token!r} is not a single token for this tokenizer")

    def score_pairs(self, pairs: Sequence[tuple[str, str]], batch_size: int = 8) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            texts = [format_qwen3_reranker_input(query, document, self.config.instruction) for query, document in batch]
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                logits = self.model(**inputs).logits[:, -1, [self.no_token_id, self.yes_token_id]]
                probs = self.torch.nn.functional.softmax(logits, dim=1)[:, 1]
            scores.extend(float(item) for item in probs.detach().cpu().tolist())
        return scores


class BM25ScoreMockReranker:
    def score_hits(self, query: str, hits: list[dict], batch_size: int = 8) -> list[float]:
        return [float(hit.get("bm25_score", 0.0)) for hit in hits]


class Qwen3HitScorer:
    def __init__(self, reranker: Qwen3Reranker):
        self.reranker = reranker

    def score_hits(self, query: str, hits: list[dict], batch_size: int = 8) -> list[float]:
        pairs = [(query, hit.get("product_text", "")) for hit in hits]
        return self.reranker.score_pairs(pairs, batch_size=batch_size)
