"""Pydantic models for the public HTTP API (also drive the OpenAPI docs at /docs)."""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {"query": "How can reward models combine human feedback with verifiable rewards?",
                 "top_k": 5, "generate_image": False}
            ]
        },
    )

    query: str = Field(
        ..., min_length=1, max_length=2000,
        validation_alias=AliasChoices("query", "question", "q"),
        description="The question to answer (aliases: question, q)",
    )
    top_k: int | None = Field(None, ge=1, le=20, description="Number of abstracts to retrieve (default: TOP_K)")
    generate_image: bool = Field(False, description="Bonus: also illustrate the answer with a generated image")


class Citation(BaseModel):
    ref: int = Field(description="Number used for inline citations in the answer, e.g. [1]")
    doc_id: str
    title: str
    url: str | None = None


class Source(BaseModel):
    ref: int
    doc_id: str
    chunk_id: str
    title: str
    text: str
    score: float = Field(description="Final fused score (Reciprocal Rank Fusion, incl. reranker when active)")
    similarity: float | None = Field(None, description="Cosine similarity to the closest search query")
    dense_rank: int | None = None
    keyword_rank: int | None = None
    hybrid_rank: int | None = Field(None, description="Rank after hybrid fusion, before reranking")
    rerank_rank: int | None = Field(None, description="Rank given by the cross-encoder")
    rerank_score: float | None = Field(None, description="Cross-encoder relevance logit")
    url: str | None = None


class QueryAnalysis(BaseModel):
    search_queries: list[str]
    keywords: list[str]
    topic: str
    queries_used: list[str]
    query_weights: list[float] = Field(default_factory=list, description="RRF weight of each query used")
    bm25_keywords: list[str] = Field(default_factory=list, description="Keywords passed to BM25")
    method: str | None = None


class GeneratedImage(BaseModel):
    url: str
    prompt: str
    provider: str
    model: str | None = None
    mime_type: str | None = None
    latency_ms: float | None = None


class Metrics(BaseModel):
    latency_ms: float | None = Field(None, description="Time until the text answer was complete")
    total_ms: float | None = Field(None, description="Total time including the optional image")
    analysis_ms: float | None = None
    retrieval_ms: float | None = None
    rerank_ms: float | None = None
    generation_ms: float | None = None
    time_to_first_token_ms: float | None = None
    image_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_second: float | None = None
    model: str
    embedding_model: str
    top_k: int | None = None
    candidates: int | None = Field(None, description="Candidates retrieved before reranking")
    retrieval_mode: str


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_context: list[str]
    sources: list[Source]
    query_analysis: QueryAnalysis
    images: list[GeneratedImage] = Field(default_factory=list)
    image_error: str | None = None
    metrics: Metrics
