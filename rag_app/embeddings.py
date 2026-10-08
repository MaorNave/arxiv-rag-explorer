"""Embedding model wrapper that applies the task prefixes many retrieval models expect."""

from __future__ import annotations

import re

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from .config import Settings

# (document prefix, query prefix) recommended by each model's authors.
_KNOWN_PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "mxbai-embed-large": ("", "Represent this sentence for searching relevant passages: "),
    "embeddinggemma": ("title: none | text: ", "task: search result | query: "),
    "qwen3-embedding": (
        "",
        "Instruct: Given a question, retrieve research paper abstracts that answer it\nQuery: ",
    ),
}


def default_prefixes(model: str) -> tuple[str, str]:
    base = model.split(":")[0].split("/")[-1].lower()
    for name, prefixes in _KNOWN_PREFIXES.items():
        if base.startswith(name):
            return prefixes
    return "", ""


class PrefixedEmbeddings(Embeddings):
    """Adds document/query prefixes before delegating to the wrapped embeddings."""

    def __init__(self, inner: Embeddings, doc_prefix: str = "", query_prefix: str = ""):
        self.inner = inner
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(self.query_prefix + text)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed several queries in a single batched call."""
        return self.inner.embed_documents([self.query_prefix + t for t in texts])


_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ns|us|µs|ms|s|m|h)")
_DURATION_UNITS = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1, "m": 60, "h": 3600}


def duration_seconds(value: str) -> int:
    """Convert an Ollama keep-alive ("30m", "1h30m", "90s", "300", "-1") to whole seconds."""
    value = value.strip().lower()
    try:
        return int(float(value))  # plain number of seconds; negative = keep the model loaded
    except ValueError:
        pass
    sign = -1 if value.startswith("-") else 1
    body = value.lstrip("+-")
    parts = _DURATION_PART.findall(body)
    if not parts or "".join(number + unit for number, unit in parts) != body:
        raise ValueError(f"Invalid duration {value!r} (examples: 30m, 1h30m, 90s, -1)")
    return sign * int(sum(float(number) * _DURATION_UNITS[unit] for number, unit in parts))


def build_embeddings(settings: Settings) -> PrefixedEmbeddings:
    doc_prefix, query_prefix = default_prefixes(settings.embed_model)
    if settings.embed_doc_prefix is not None:
        doc_prefix = settings.embed_doc_prefix
    if settings.embed_query_prefix is not None:
        query_prefix = settings.embed_query_prefix
    inner = OllamaEmbeddings(
        model=settings.embed_model,
        base_url=settings.ollama_base_url,
        keep_alive=duration_seconds(settings.llm_keep_alive),
    )
    return PrefixedEmbeddings(inner, doc_prefix, query_prefix)
