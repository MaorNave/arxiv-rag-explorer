"""Cross-encoder reranking of the hybrid-retrieval candidates (local ONNX model).

Bi-encoders and BM25 score the question and each abstract independently; a
cross-encoder reads them *together*, which is far more precise but too slow for the
whole collection. The pipeline therefore retrieves ~20 candidates broadly (hybrid
search + LLM rewrites at full weight) and lets the cross-encoder pick the top-k.

The final order blends the cross-encoder rank with the hybrid rank through
Reciprocal Rank Fusion: score = w / (60 + rank_ce) + 1 / (60 + rank_hybrid).
On the bundled benchmark (scripts/eval_retrieval.py) this beats both the hybrid
ranking alone and the pure cross-encoder ordering.

The model (cross-encoder/ms-marco-MiniLM-L6-v2, Apache-2.0, ~90 MB) is downloaded
once from Hugging Face into STORAGE_DIR/models and then runs fully offline on CPU
with ONNX Runtime.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

RRF_K = 60


class CrossEncoderReranker:
    def __init__(self, model: str, cache_dir: Path, max_tokens: int = 384):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.max_tokens = max_tokens
        self._session = None
        self._tokenizer = None
        self._inputs: set[str] = set()
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._session is not None

    def _fetch(self, filename: str) -> str:
        from huggingface_hub import hf_hub_download

        try:  # offline-first: reuse the cached copy without touching the network
            return hf_hub_download(self.model, filename, cache_dir=self.cache_dir, local_files_only=True)
        except Exception:
            return hf_hub_download(self.model, filename, cache_dir=self.cache_dir)

    def load(self) -> None:
        """Download (first run only) and load the model. Raises if that is impossible."""
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = self._fetch("onnx/model.onnx")
        tokenizer = Tokenizer.from_file(self._fetch("tokenizer.json"))
        tokenizer.enable_truncation(max_length=self.max_tokens)
        tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.log_severity_level = 3
        session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
        with self._lock:
            self._tokenizer = tokenizer
            self._session = session
            self._inputs = {i.name for i in session.get_inputs()}
        log.info("Reranker %s loaded", self.model)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance logits for (query, passage) pairs; higher is more relevant."""
        if not passages:
            return []
        with self._lock:
            if self._session is None:
                raise RuntimeError("Reranker is not loaded")
            encodings = self._tokenizer.encode_batch([(query, p) for p in passages])
            feeds = {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            }
            if "token_type_ids" in self._inputs:
                feeds["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
            logits = self._session.run(None, feeds)[0]
        return np.asarray(logits, dtype=np.float32).reshape(len(passages), -1)[:, 0].tolist()


def blend_rankings(scores: Sequence[float], weight: float = 2.0, rrf_k: int = RRF_K) -> list[tuple[int, int, float]]:
    """Fuse the cross-encoder order with the incoming (hybrid) order.

    ``scores[i]`` belongs to the candidate at hybrid rank i + 1. Returns
    ``(candidate_index, cross_encoder_rank, fused_score)`` sorted best first.
    """
    by_ce = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ce_rank = {index: rank for rank, index in enumerate(by_ce, start=1)}
    fused = [(i, ce_rank[i], weight / (rrf_k + ce_rank[i]) + 1.0 / (rrf_k + i + 1)) for i in range(len(scores))]
    return sorted(fused, key=lambda item: item[2], reverse=True)
