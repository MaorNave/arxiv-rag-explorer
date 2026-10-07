"""Hybrid retrieval: dense vectors (Chroma) + BM25 keywords (SQLite FTS5), fused with RRF."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .dataset import arxiv_url
from .keyword_index import build_match_query


@dataclass
class Passage:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float = 0.0  # fused Reciprocal Rank Fusion score
    similarity: float | None = None  # best cosine similarity to any search query
    dense_rank: int | None = None  # best rank in the dense result lists
    keyword_rank: int | None = None  # rank in the BM25 result list

    def to_dict(self) -> dict:
        data = asdict(self)
        data["score"] = round(self.score, 5)
        if self.similarity is not None:
            data["similarity"] = round(self.similarity, 4)
        data["url"] = arxiv_url(self.doc_id)
        return data


class HybridRetriever(BaseRetriever):
    """LangChain retriever over an :class:`ActiveIndex`.

    ``search()`` accepts the user's question plus optional query rewrites and
    keywords. Every query is searched densely, the primary query (weight >= 1)
    plus the keywords drive one BM25 search, and all ranked lists are merged with
    weighted Reciprocal Rank Fusion: score = sum(weight / (rrf_k + rank)).
    Down-weighting rewrites keeps a drifting LLM rewrite from outvoting the
    question itself. Results are collapsed to the best chunk per paper so
    citations stay unique.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: Any  # ActiveIndex (kept as Any so pydantic does not copy/validate it)
    k: int = 5
    fetch_k: int = 20
    rrf_k: int = 60
    mode: str = "hybrid"  # "hybrid" | "dense" | "keyword" (the last two exist for ablations)
    keyword_weight: float = 1.0

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [
            Document(page_content=p.text, metadata={k: v for k, v in p.to_dict().items() if k != "text"})
            for p in self.search([query])
        ]

    def search(
        self,
        queries: list[str],
        keywords: list[str] | None = None,
        k: int | None = None,
        *,
        weights: list[float] | None = None,
    ) -> list[Passage]:
        k = k or self.k
        keywords = keywords or []
        weighted: dict[str, float] = {}
        for i, query in enumerate(queries):
            query = (query or "").strip()
            if query and query not in weighted:
                weighted[query] = weights[i] if weights else 1.0
        if not weighted:
            return []
        texts = list(weighted)

        fused: dict[str, Passage] = {}
        query_vectors = np.asarray(self.index.embeddings.embed_queries(texts), dtype=np.float32)

        # Dense: one ranked list per query, weighted.
        for vector, weight in zip(query_vectors, weighted.values(), strict=True) if self.mode != "keyword" else []:
            hits = self.index.vectorstore.similarity_search_by_vector_with_relevance_scores(
                vector.tolist(), k=self.fetch_k
            )
            for rank, (doc, distance) in enumerate(hits, start=1):
                passage = self._passage(fused, doc.id, doc.metadata, doc.page_content)
                passage.score += weight / (self.rrf_k + rank)
                similarity = 1.0 - float(distance)  # cosine distance -> similarity
                passage.similarity = similarity if passage.similarity is None else max(passage.similarity, similarity)
                passage.dense_rank = rank if passage.dense_rank is None else min(passage.dense_rank, rank)

        # Sparse: one BM25 list from the primary query (the question) + keywords.
        primary = [text for text, weight in weighted.items() if weight >= 1.0] or texts
        match = build_match_query(primary + keywords, phrases=keywords) if self.mode != "dense" else None
        if match:
            for rank, hit in enumerate(self.index.keywords.search(match, self.fetch_k), start=1):
                passage = self._passage(fused, hit.chunk_id, {"doc_id": hit.doc_id, "title": hit.title}, hit.text)
                passage.score += self.keyword_weight / (self.rrf_k + rank)
                passage.keyword_rank = rank

        # Keep the best chunk per document, then the top-k documents.
        best: dict[str, Passage] = {}
        for passage in fused.values():
            current = best.get(passage.doc_id)
            if current is None or passage.score > current.score:
                best[passage.doc_id] = passage
        ranked = sorted(best.values(), key=lambda p: p.score, reverse=True)[:k]

        self._fill_missing_similarity(ranked, query_vectors)
        return ranked

    @staticmethod
    def _passage(fused: dict[str, Passage], chunk_id: str | None, metadata: dict, text: str) -> Passage:
        doc_id = str(metadata.get("doc_id", ""))
        chunk_id = chunk_id or f"{doc_id}#0"
        passage = fused.get(chunk_id)
        if passage is None:
            passage = Passage(chunk_id=chunk_id, doc_id=doc_id, title=str(metadata.get("title", doc_id)), text=text)
            fused[chunk_id] = passage
        return passage

    def _fill_missing_similarity(self, passages: list[Passage], query_vectors: np.ndarray) -> None:
        """Keyword-only hits have no cosine score yet; compute it from their stored vectors."""
        missing = [p for p in passages if p.similarity is None]
        if not missing:
            return
        stored = self.index.vectorstore.get(ids=[p.chunk_id for p in missing], include=["embeddings"])
        vectors = {cid: np.asarray(vec, dtype=np.float32) for cid, vec in zip(stored["ids"], stored["embeddings"], strict=True)}
        q_norms = np.linalg.norm(query_vectors, axis=1) + 1e-12
        for passage in missing:
            vec = vectors.get(passage.chunk_id)
            if vec is not None:
                sims = (query_vectors @ vec) / (q_norms * (np.linalg.norm(vec) + 1e-12))
                passage.similarity = float(sims.max())
