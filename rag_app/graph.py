"""The RAG pipeline as a LangGraph state machine.

    START -> understand_query -> retrieve -> rerank -+-> generate_answer --+-> finalize -> END
                                                     +-> generate_image ---+   (if requested)

    With a local image model the image is generated after the answer instead:
    ... -> rerank -> generate_answer -> generate_image -> finalize

* understand_query  LLM (structured output) rewrites the question into English search
                    queries + keywords + topic; falls back to a heuristic plan.
* retrieve          hybrid dense + BM25 retrieval fused with weighted Reciprocal Rank Fusion
                    (~20 broad candidates when the reranker is active, top-k otherwise).
* rerank            local cross-encoder (ONNX) re-scores the candidates against the
                    question and keeps the top-k (pass-through if no reranker is loaded).
* generate_answer   grounded, cited answer streamed token by token from the local LLM.
* generate_image    optional image. External APIs run in the same super-step as
                    generate_answer (concurrently); a local model runs right after it, so
                    it never competes with the LLM. Either way the text is never delayed.
* finalize          assembles the response JSON (answer, citations, context, metrics).

Nodes publish progress through LangGraph's custom stream (``get_stream_writer``),
which the /stream endpoint forwards to the browser as Server-Sent Events.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from .config import Settings
from .dataset import arxiv_url
from .images import ImageGenerator, build_image_prompt
from .keyword_index import _tokens
from .prompts import (
    ANALYSIS_PROMPT,
    ANSWER_PROMPT,
    EMPTY_ANSWER,
    NO_CONTEXT_ANSWER,
    format_context,
)
from .reranker import CrossEncoderReranker, blend_rankings
from .retriever import HybridRetriever

log = logging.getLogger(__name__)


def _merge(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


class RAGState(TypedDict, total=False):
    # inputs
    question: str
    top_k: int
    generate_image: bool
    image_after_answer: bool
    started_at: float
    # produced by the nodes
    plan: dict[str, Any]
    candidates: list[dict[str, Any]]
    passages: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]
    llm: dict[str, Any]
    images: list[dict[str, Any]]
    image_error: str | None
    timings: Annotated[dict[str, float], _merge]  # parallel branches merge their timings
    response: dict[str, Any]


@dataclass
class PipelineDeps:
    settings: Settings
    llm: BaseChatModel
    get_retriever: Callable[[], HybridRetriever]
    analyzer: Runnable | None = None
    image_generator: ImageGenerator | None = None
    reranker: CrossEncoderReranker | None = None

    @property
    def reranking(self) -> bool:
        return self.reranker is not None and self.reranker.ready


# ---------------------------------------------------------------- helpers
def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _clean_list(values: Any, limit: int) -> list[str]:
    items: list[str] = []
    for value in values or []:
        text = " ".join(str(value).split()).strip(" .,;")
        if text and text.lower() not in (i.lower() for i in items):
            items.append(text)
    return items[:limit]


def heuristic_plan(question: str) -> dict[str, Any]:
    terms = list(dict.fromkeys(_tokens(question)))
    return {
        "search_queries": [question.strip()],
        "keywords": terms[:6],
        "topic": " ".join(terms[:6]) or question.strip()[:80],
        "method": "heuristic",
    }


def _mostly_latin(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return not letters or sum(c.isascii() for c in letters) / len(letters) >= 0.6


REWRITE_WEIGHT = 0.25  # rewrites add recall but must not outvote the question (see scripts/eval_retrieval.py)


def fusion_weights(settings: Settings, reranking: bool) -> tuple[float, float]:
    """(rewrite_weight, keyword_weight) for weighted RRF.

    With a cross-encoder to restore precision, retrieval favours recall (every query at
    full weight). Without it, rewrites are damped and BM25 is boosted, which the
    benchmark showed is the best compromise. Explicit settings always win.
    """
    rewrite = settings.rewrite_weight if settings.rewrite_weight is not None else (1.0 if reranking else REWRITE_WEIGHT)
    keyword = settings.keyword_weight if settings.keyword_weight is not None else (1.0 if reranking else 1.5)
    return rewrite, keyword


def _related(token: str, others: set[str]) -> bool:
    return any(token == o or (len(token) >= 5 and len(o) >= 5 and token[:5] == o[:5]) for o in others)


def anchored_topic(question: str, topic: str) -> str:
    """Keep the planner's topic only if it shares a word with the question (Latin script);
    small models sometimes "explain" unknown acronyms with an invented topic."""
    question_tokens = set(_tokens(question))
    if topic and (not _mostly_latin(question) or any(_related(t, question_tokens) for t in _tokens(topic))):
        return topic
    return heuristic_plan(question)["topic"]


def retrieval_inputs(
    question: str, plan: dict[str, Any], rewrite_weight: float = REWRITE_WEIGHT
) -> tuple[list[str], list[float], list[str]]:
    """Queries, RRF weights and BM25 keywords for the retriever.

    * The original question anchors retrieval (weight 1.0); LLM rewrites add recall at
      ``rewrite_weight``. A question in a non-Latin script is left out: its English
      rewrites are what the embedder and BM25 can actually use.
    * BM25 only receives keywords whose words all occur in the question, because the planner
      sometimes adds associated (or guessed) terms, e.g. "latent diffusion" for a question
      about "diffusion", whose phrase boost would pull exact matching off-topic.
      For non-Latin questions all keywords are kept (they are the translations).
    """
    question = question.strip()
    rewrites = [q.strip() for q in plan.get("search_queries") or [] if q.strip()]
    if _mostly_latin(question) or not rewrites:
        pairs = [(question, 1.0)] + [(q, rewrite_weight) for q in rewrites]
        question_tokens = set(_tokens(question))
        keywords = [
            kw for kw in plan.get("keywords") or []
            if _tokens(kw) and all(_related(t, question_tokens) for t in _tokens(kw))
        ]
    else:
        pairs = [(q, 1.0) for q in rewrites]
        keywords = list(plan.get("keywords") or [])
    seen: dict[str, float] = {}
    for query, weight in pairs:
        if query and query.lower() not in (s.lower() for s in seen):
            seen[query] = weight
    queries = list(seen)[:4]
    return queries, [seen[q] for q in queries], keywords


_CITATION_RE = re.compile(r"\[\s*(?:abstracts?|sources?|docs?|passages?|refs?)?\s*(\d+(?:\s*[-–,;]\s*\d+)*)\s*\]", re.I)
_LABELLED_RE = re.compile(r"\[\s*(?:abstracts?|sources?|docs?|passages?|refs?)\s*(\d+)\s*\]", re.I)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def clean_answer(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _LABELLED_RE.sub(r"[\1]", text)  # "[Abstract 2]" -> "[2]"
    return text.strip()


def cited_refs(answer: str, passages: list[dict[str, Any]]) -> list[int]:
    """Passage numbers cited in the answer, in order of first appearance."""
    n = len(passages)
    refs: list[int] = []
    for match in _CITATION_RE.finditer(answer):
        for part in re.split(r"\s*[,;]\s*", match.group(1)):
            bounds = re.split(r"\s*[-–]\s*", part)
            try:
                lo, hi = int(bounds[0]), int(bounds[-1])
            except ValueError:
                continue
            for ref in range(lo, min(hi, lo + 20) + 1):
                if 1 <= ref <= n and ref not in refs:
                    refs.append(ref)
    for ref, passage in enumerate(passages, start=1):  # models sometimes cite by doc id
        if ref not in refs and passage["doc_id"] in answer:
            refs.append(ref)
    return refs


def build_citations(answer: str, passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs = cited_refs(answer, passages) or list(range(1, len(passages) + 1))
    return [
        {
            "ref": ref,
            "doc_id": passages[ref - 1]["doc_id"],
            "title": passages[ref - 1]["title"],
            "url": arxiv_url(passages[ref - 1]["doc_id"]),
        }
        for ref in refs
    ]


# ---------------------------------------------------------------- graph
def build_graph(deps: PipelineDeps):
    settings = deps.settings

    async def understand_query(state: RAGState, config: RunnableConfig) -> dict:
        write = get_stream_writer()
        write({"event": "step", "data": {"step": "understand", "status": "running"}})
        started = time.perf_counter()
        question = state["question"]
        plan: dict[str, Any] | None = None
        if settings.query_rewrite and deps.analyzer is not None:
            try:
                messages = ANALYSIS_PROMPT.format_messages(question=question)
                result = await asyncio.wait_for(
                    deps.analyzer.ainvoke(messages, config=config), timeout=settings.analysis_timeout_s
                )
                plan = {
                    "search_queries": _clean_list(result.search_queries, 2),
                    "keywords": _clean_list(result.keywords, 5),
                    "topic": " ".join(result.topic.split())[:120],
                    "method": "llm",
                }
            except Exception as exc:  # malformed JSON, timeout, ... -> heuristic plan
                log.warning("Query analysis failed (%s: %s); using heuristic plan", type(exc).__name__, exc)
        if not plan or not plan["search_queries"]:
            plan = heuristic_plan(question)
        plan["topic"] = anchored_topic(question, plan.get("topic", ""))
        rewrite_weight, _ = fusion_weights(settings, deps.reranking)
        queries, weights, keywords = retrieval_inputs(question, plan, rewrite_weight)
        plan.update(queries_used=queries, query_weights=weights, bm25_keywords=keywords)
        elapsed = _ms(started)
        write({"event": "analysis", "data": {**plan, "ms": elapsed}})
        # Decided once per request so that a provider fallback cannot change the routing mid-way.
        after = bool(state.get("generate_image") and deps.image_generator and deps.image_generator.runs_after_answer)
        return {"plan": plan, "timings": {"analysis_ms": elapsed}, "image_after_answer": after}

    async def retrieve(state: RAGState) -> dict:
        write = get_stream_writer()
        write({"event": "step", "data": {"step": "retrieve", "status": "running"}})
        started = time.perf_counter()
        plan = state["plan"]
        retriever = deps.get_retriever()
        _, retriever.keyword_weight = fusion_weights(settings, deps.reranking)
        limit = max(settings.rerank_candidates, state["top_k"]) if deps.reranking else state["top_k"]
        found = await asyncio.to_thread(
            retriever.search, plan["queries_used"], plan["bm25_keywords"], limit,
            weights=plan["query_weights"],
        )
        candidates = [p.to_dict() for p in found]
        elapsed = _ms(started)
        write({"event": "step", "data": {"step": "retrieve", "status": "done", "ms": elapsed,
                                         "candidates": len(candidates)}})
        return {"candidates": candidates, "timings": {"retrieval_ms": elapsed}}

    async def rerank(state: RAGState) -> dict:
        write = get_stream_writer()
        candidates = state.get("candidates") or []
        top_k = state["top_k"]
        started = time.perf_counter()
        chosen: list[dict[str, Any]] | None = None
        if deps.reranking and len(candidates) > 1:
            write({"event": "step", "data": {"step": "rerank", "status": "running"}})
            plan = state["plan"]
            question = state["question"]
            # The cross-encoder is English-only: use the planner's English rewrite for other scripts.
            query = question if _mostly_latin(question) else (plan.get("search_queries") or [question])[0]
            texts = [f"{c['title']}. {c['text']}" for c in candidates]
            try:
                scores = await asyncio.to_thread(deps.reranker.score, query, texts)
                chosen = []
                for index, ce_rank, fused in blend_rankings(scores, settings.rerank_weight)[:top_k]:
                    chosen.append({**candidates[index], "hybrid_rank": index + 1, "rerank_rank": ce_rank,
                                   "rerank_score": round(scores[index], 3), "score": round(fused, 5)})
            except Exception as exc:  # never fail the answer because of the reranker
                log.warning("Reranking failed (%s); keeping the hybrid order", exc)
        reranked = chosen is not None
        if chosen is None:
            chosen = [{**c, "hybrid_rank": i} for i, c in enumerate(candidates[:top_k], start=1)]
        passages = [{"ref": ref, **p} for ref, p in enumerate(chosen, start=1)]
        elapsed = _ms(started)
        write({"event": "retrieval", "data": {
            "passages": passages, "candidates": len(candidates), "reranked": reranked, "ms": elapsed,
        }})
        return {"passages": passages, "timings": {"rerank_ms": elapsed if reranked else None}}

    def route_after_retrieval(state: RAGState) -> list[str]:
        targets = ["generate_answer"]
        if state.get("generate_image") and deps.image_generator is not None and not state.get("image_after_answer"):
            targets.append("generate_image")  # external API: run concurrently with the answer
        return targets

    def route_after_answer(state: RAGState) -> str:
        if state.get("generate_image") and deps.image_generator is not None and state.get("image_after_answer"):
            return "generate_image"  # local model: start once the LLM is done
        return "finalize"

    async def generate_answer(state: RAGState, config: RunnableConfig) -> dict:
        write = get_stream_writer()
        write({"event": "step", "data": {"step": "generate", "status": "running"}})
        started = time.perf_counter()
        passages = state.get("passages") or []
        stats: dict[str, Any] = {"model": settings.llm_model}

        if not passages:
            answer = NO_CONTEXT_ANSWER
            write({"event": "token", "data": {"text": answer}})
        else:
            messages = ANSWER_PROMPT.format_messages(
                question=state["question"], context=format_context(passages), count=len(passages)
            )
            parts: list[str] = []
            first_token: float | None = None  # first answer token (what the user waits for)
            first_output: float | None = None  # first token of any kind, incl. reasoning (for tokens/s)
            aggregate = None
            async for chunk in deps.llm.astream(messages, config=config):
                reasoning = chunk.additional_kwargs.get("reasoning_content")
                text = chunk.text
                if (reasoning or text) and first_output is None:
                    first_output = time.perf_counter()
                if reasoning:
                    write({"event": "thinking", "data": {"text": reasoning}})
                if text:
                    if first_token is None:
                        first_token = time.perf_counter()
                    parts.append(text)
                    write({"event": "token", "data": {"text": text}})
                aggregate = chunk if aggregate is None else aggregate + chunk
            answer = clean_answer("".join(parts)) or EMPTY_ANSWER
            usage = (getattr(aggregate, "usage_metadata", None) or {}) if aggregate is not None else {}
            finished = time.perf_counter()
            completion = usage.get("output_tokens")  # Ollama counts reasoning tokens here too
            stats.update(
                prompt_tokens=usage.get("input_tokens"),
                completion_tokens=completion,
                time_to_first_token_ms=round((first_token - started) * 1000, 1) if first_token else None,
                tokens_per_second=(
                    round(completion / (finished - first_output), 1)
                    if completion and first_output and finished > first_output else None
                ),
            )

        citations = build_citations(answer, passages) if passages else []
        elapsed = _ms(started)
        answer_latency = round((time.perf_counter() - state["started_at"]) * 1000, 1)
        write({"event": "answer", "data": {"answer": answer, "citations": citations, "ms": elapsed}})
        return {
            "answer": answer,
            "citations": citations,
            "llm": stats,
            "timings": {"generation_ms": elapsed, "answer_latency_ms": answer_latency},
        }

    async def generate_image(state: RAGState) -> dict:
        write = get_stream_writer()
        write({"event": "step", "data": {"step": "image", "status": "running"}})
        started = time.perf_counter()
        prompt = build_image_prompt(state["question"], state.get("plan"))
        # Each provider has its own timeout; this is only a safety net for the whole fallback chain.
        budget = settings.local_image_timeout_s + 2 * settings.image_timeout_s + 30
        try:
            image = await asyncio.wait_for(deps.image_generator.generate(prompt), timeout=budget)
            images, error = [image], None
        except Exception as exc:
            log.warning("Image generation failed: %s", exc)
            images, error = [], str(exc) or type(exc).__name__
        elapsed = _ms(started)
        write({"event": "image", "data": {"images": images, "error": error, "prompt": prompt, "ms": elapsed}})
        return {"images": images, "image_error": error, "timings": {"image_ms": elapsed}}

    async def finalize(state: RAGState) -> dict:
        passages = state.get("passages") or []
        timings = state.get("timings") or {}
        stats = state.get("llm") or {}
        plan = state.get("plan") or {}
        response = {
            "answer": state.get("answer", ""),
            "citations": state.get("citations", []),
            "retrieved_context": [p["text"] for p in passages],
            "sources": passages,
            "query_analysis": {
                "search_queries": plan.get("search_queries", []),
                "keywords": plan.get("keywords", []),
                "topic": plan.get("topic", ""),
                "queries_used": plan.get("queries_used", []),
                "query_weights": plan.get("query_weights", []),
                "bm25_keywords": plan.get("bm25_keywords", []),
                "method": plan.get("method"),
            },
            "images": state.get("images", []),
            "image_error": state.get("image_error"),
            "metrics": {
                "latency_ms": timings.get("answer_latency_ms"),
                "total_ms": round((time.perf_counter() - state["started_at"]) * 1000, 1),
                "analysis_ms": timings.get("analysis_ms"),
                "retrieval_ms": timings.get("retrieval_ms"),
                "rerank_ms": timings.get("rerank_ms"),
                "generation_ms": timings.get("generation_ms"),
                "time_to_first_token_ms": stats.get("time_to_first_token_ms"),
                "image_ms": timings.get("image_ms"),
                "prompt_tokens": stats.get("prompt_tokens"),
                "completion_tokens": stats.get("completion_tokens"),
                "tokens_per_second": stats.get("tokens_per_second"),
                "model": settings.llm_model,
                "embedding_model": settings.embed_model,
                "top_k": state.get("top_k"),
                "retrieval_mode": "hybrid: dense (Chroma) + BM25 (SQLite FTS5), Reciprocal Rank Fusion"
                + (f" + cross-encoder rerank ({deps.reranker.model})" if deps.reranking else ""),
                "candidates": len(state.get("candidates") or []),
            },
        }
        return {"response": response}

    graph = StateGraph(RAGState)
    graph.add_node("understand_query", understand_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("generate_image", generate_image)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "understand_query")
    graph.add_edge("understand_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges("rerank", route_after_retrieval, ["generate_answer", "generate_image"])
    graph.add_conditional_edges("generate_answer", route_after_answer, ["generate_image", "finalize"])
    graph.add_edge("generate_image", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(name="arxiv-rag")


def initial_state(question: str, top_k: int, generate_image: bool) -> RAGState:
    return {
        "question": question.strip(),
        "top_k": top_k,
        "generate_image": generate_image,
        "started_at": time.perf_counter(),
        "timings": {},
    }
