"""Shared fixtures: tiny datasets, offline fake models and isolated settings.

The suite never talks to Ollama or the internet: embeddings are a deterministic
bag-of-words hash (so lexical overlap still means "similar") and the chat model is
LangChain's GenericFakeChatModel.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from rag_app.config import Settings

RECORDS = [
    {"id": "2509.00001v1", "title": "RLBFF: Binary Flexible Feedback for reward models",
     "abstract": "We propose RLBFF, which combines human feedback with verifiable rewards so that reward models "
                 "capture nuanced aspects of response quality.", "authors": "A. Author", "categories": "cs.CL"},
    {"id": "2509.00002v1", "title": "Few-step diffusion distillation",
     "abstract": "We distill a diffusion model into a student that generates images in only four sampling steps "
                 "while keeping image quality."},
    {"id": "2509.00003v1", "title": "Detecting hallucinations in language models",
     "abstract": "Large language models hallucinate. We detect hallucinations from hidden layer signals and "
                 "benchmark detectors on question answering."},
    {"id": "2509.00004v1", "title": "Gaussian splatting avatars",
     "abstract": "Gaussian splatting reconstructs animatable 3D human avatars from monocular video."},
    {"id": "2509.00005v1", "title": "Graph neural networks for molecules",
     "abstract": "Graph neural networks predict molecular properties using message passing over atoms and bonds."},
]


def write_jsonl(path: Path, records: list[dict], extra_lines: list[str] | None = None) -> Path:
    lines = [json.dumps(r) for r in records] + (extra_lines or [])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings: texts sharing words get similar vectors."""

    def __init__(self, size: int = 64):
        self.size = size
        self.calls = 0

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.size
        for token in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.size] += 1.0 if (h >> 8) % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def fake_llm(answer: str = "Reward models can mix human feedback with verifiable rewards [1]. "
                           "Distillation also helps [2].") -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=itertools.cycle([AIMessage(content=answer)]))


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    extra = [
        "",  # blank line
        "{not valid json",  # malformed
        json.dumps({"id": "2509.00001v1", "title": "duplicate", "abstract": "duplicate id"}),
        json.dumps({"id": "no-abstract", "title": "missing text"}),
    ]
    return write_jsonl(data_dir / "papers.jsonl", RECORDS, extra)


@pytest.fixture
def settings(tmp_path: Path, dataset: Path) -> Settings:
    return Settings(
        data_path=str(dataset),
        storage_dir=tmp_path / "storage",
        query_rewrite=False,
        reranker_model="none",  # tests never download models
        image_provider="none",
        watch_dataset=False,
        embed_batch_size=2,
        chunk_size=2000,
        chunk_overlap=100,
        top_k=3,
        fetch_k=5,
    )


@pytest.fixture
def embeddings() -> HashEmbeddings:
    return HashEmbeddings()
