import asyncio

import pytest

from rag_app.embeddings import PrefixedEmbeddings
from rag_app.graph import (
    PipelineDeps,
    anchored_topic,
    build_citations,
    build_graph,
    cited_refs,
    clean_answer,
    initial_state,
    retrieval_inputs,
)
from rag_app.images import build_image_prompt
from rag_app.index_manager import IndexManager
from rag_app.prompts import NO_CONTEXT_ANSWER
from rag_app.reranker import blend_rankings
from rag_app.retriever import HybridRetriever

from .conftest import fake_llm

PASSAGES = [{"doc_id": f"2509.0000{i}v1", "title": f"Paper {i}", "text": "..."} for i in range(1, 5)]


def test_cited_refs_formats():
    assert cited_refs("A [1] and B [3][2].", PASSAGES) == [1, 3, 2]
    assert cited_refs("See [1, 4] and [2-3].", PASSAGES) == [1, 4, 2, 3]
    assert cited_refs("Out of range [9] ignored; by id 2509.00004v1.", PASSAGES) == [4]
    assert cited_refs(clean_answer("Per [Abstract 2] and [source 1]."), PASSAGES) == [2, 1]


def test_build_citations_falls_back_to_all_passages():
    cits = build_citations("No markers here.", PASSAGES[:2])
    assert [c["ref"] for c in cits] == [1, 2]
    assert cits[0] == {"ref": 1, "doc_id": "2509.00001v1", "title": "Paper 1", "url": "https://arxiv.org/abs/2509.00001v1"}


def test_clean_answer_strips_reasoning_tags():
    assert clean_answer("<think>hmm</think>\nFinal [1].") == "Final [1]."


def test_retrieval_inputs_weights_and_keyword_filtering():
    plan = {"search_queries": ["Reinforcement learning backprop", "RLBFF method"], "keywords": ["RLBFF", "policy gradient"]}
    queries, weights, keywords = retrieval_inputs("What is RLBFF?", plan, rewrite_weight=0.25)
    assert queries == ["What is RLBFF?", "Reinforcement learning backprop", "RLBFF method"]
    assert weights == [1.0, 0.25, 0.25]
    assert keywords == ["RLBFF"]  # the guessed "policy gradient" is not in the question

    plan = {"search_queries": [], "keywords": ["diffusion models", "latent diffusion"]}
    assert retrieval_inputs("How do diffusion models generate images?", plan)[2] == ["diffusion models"]

    hebrew_plan = {"search_queries": ["few-step diffusion distillation"], "keywords": ["diffusion", "distillation"]}
    queries, weights, keywords = retrieval_inputs("איך מזקקים מודלי דיפוזיה?", hebrew_plan)
    assert queries == ["few-step diffusion distillation"] and weights == [1.0]
    assert keywords == ["diffusion", "distillation"]  # translations are kept


def test_topic_is_anchored_to_the_question():
    assert anchored_topic("What is RLBFF?", "Reinforcement Learning with Backpropagation") == "rlbff"
    assert anchored_topic("How do LLMs hallucinate?", "hallucination in LLMs") == "hallucination in LLMs"
    assert anchored_topic("איך מזקקים מודלי דיפוזיה?", "diffusion distillation") == "diffusion distillation"


def test_image_prompt_has_no_llm_dependency():
    prompt = build_image_prompt("How do diffusion models work?", {"topic": "diffusion models", "keywords": ["denoising"]})
    assert "diffusion models" in prompt and "denoising" in prompt and "No text" in prompt


@pytest.fixture
def index(settings, embeddings, dataset):
    return IndexManager(settings, PrefixedEmbeddings(embeddings, model="fake")).load_or_build(dataset)


class FakeImages:
    def __init__(self, fail=False, delay=0.0, runs_after_answer=False):
        self.fail, self.delay, self.runs_after_answer = fail, delay, runs_after_answer

    async def generate(self, prompt):
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("quota exceeded")
        return {"url": "/generated/x.png", "prompt": prompt, "provider": "fake", "model": None, "latency_ms": 1}


def run_graph(settings, index, *, question="What is RLBFF?", image=None, generate_image=False, retriever=None):
    deps = PipelineDeps(
        settings=settings,
        llm=fake_llm(),
        get_retriever=retriever or (lambda: HybridRetriever(index=index, k=3, fetch_k=5)),
        image_generator=image,
    )
    graph = build_graph(deps)
    return asyncio.run(graph.ainvoke(initial_state(question, 3, generate_image)))["response"]


def test_graph_produces_the_assignment_schema(settings, index):
    response = run_graph(settings, index)
    assert response["answer"].startswith("Reward models")
    assert [c["ref"] for c in response["citations"]] == [1, 2]
    assert response["citations"][0]["doc_id"] == "2509.00001v1"
    assert all({"doc_id", "title"} <= set(c) for c in response["citations"])
    assert len(response["retrieved_context"]) == 3 and all(isinstance(t, str) for t in response["retrieved_context"])
    assert response["query_analysis"]["method"] == "heuristic"  # QUERY_REWRITE disabled in tests
    assert response["metrics"]["latency_ms"] > 0 and response["images"] == []


def test_graph_streams_events_in_order(settings, index):
    deps = PipelineDeps(settings=settings, llm=fake_llm(), get_retriever=lambda: HybridRetriever(index=index, k=3))
    graph = build_graph(deps)

    async def collect():
        events = []
        async for mode, chunk in graph.astream(initial_state("What is RLBFF?", 3, False), stream_mode=["custom", "updates"]):
            if mode == "custom":
                events.append(chunk["event"])
        return events

    events = asyncio.run(collect())
    assert events.index("analysis") < events.index("retrieval") < events.index("token") < events.index("answer")
    assert events.count("token") > 3  # streamed, not one blob


def test_image_branch_success_and_failure(settings, index):
    ok = run_graph(settings, index, image=FakeImages(), generate_image=True)
    assert ok["images"][0]["url"] == "/generated/x.png" and ok["image_error"] is None

    failed = run_graph(settings, index, image=FakeImages(fail=True), generate_image=True)
    assert failed["images"] == [] and "quota" in failed["image_error"]
    assert failed["answer"]  # the text answer is unaffected


def test_slow_image_does_not_delay_the_answer(settings, index):
    deps = PipelineDeps(settings=settings, llm=fake_llm(), get_retriever=lambda: HybridRetriever(index=index, k=3),
                        image_generator=FakeImages(delay=1.0))
    graph = build_graph(deps)

    async def timings():
        loop = asyncio.get_running_loop()
        start, seen = loop.time(), {}
        async for _mode, chunk in graph.astream(initial_state("What is RLBFF?", 3, True), stream_mode=["custom"]):
            seen.setdefault(chunk["event"], loop.time() - start)
        return seen

    seen = asyncio.run(timings())
    assert seen["answer"] < 0.9 <= seen["image"]  # answer finished while the image was still generating


def test_no_context_answer(settings, index):
    class Empty:
        def search(self, *args, **kwargs):
            return []

    response = run_graph(settings, index, retriever=lambda: Empty())
    assert response["answer"] == NO_CONTEXT_ANSWER
    assert response["citations"] == [] and response["retrieved_context"] == []


def test_blend_rankings_mixes_cross_encoder_and_hybrid_order():
    order = blend_rankings([0.1, 0.9, 0.5], weight=2.0)
    assert [index for index, _, _ in order] == [1, 0, 2]
    assert [ce_rank for _, ce_rank, _ in order] == [1, 3, 2]


class FakeReranker:
    model = "fake-cross-encoder"
    ready = True

    def __init__(self, prefer):
        self.prefer = prefer
        self.queries = []

    def score(self, query, texts):
        self.queries.append(query)
        return [10.0 if self.prefer in text else 0.0 for text in texts]


def test_rerank_node_promotes_cross_encoder_choice(settings, index):
    hybrid_only = run_graph(settings, index)
    assert all(s["doc_id"] != "2509.00005v1" for s in hybrid_only["sources"])  # molecules paper not in top-3

    reranker = FakeReranker(prefer="molecular")
    deps = PipelineDeps(settings=settings, llm=fake_llm(), reranker=reranker,
                        get_retriever=lambda: HybridRetriever(index=index, k=3, fetch_k=5))
    response = asyncio.run(build_graph(deps).ainvoke(initial_state("What is RLBFF?", 3, False)))["response"]
    sources = {s["doc_id"]: s for s in response["sources"]}
    assert "2509.00005v1" in sources and sources["2509.00005v1"]["rerank_rank"] == 1
    assert response["sources"][0]["doc_id"] == "2509.00001v1"  # the hybrid #1 survives the blend
    assert response["metrics"]["candidates"] == 5 and response["metrics"]["rerank_ms"] is not None
    assert "cross-encoder" in response["metrics"]["retrieval_mode"]
    assert reranker.queries == ["What is RLBFF?"]


def test_local_image_model_runs_after_the_answer(settings, index):
    deps = PipelineDeps(settings=settings, llm=fake_llm(), get_retriever=lambda: HybridRetriever(index=index, k=3),
                        image_generator=FakeImages(runs_after_answer=True))
    graph = build_graph(deps)

    async def events():
        seen = []
        async for _mode, chunk in graph.astream(initial_state("What is RLBFF?", 3, True), stream_mode=["custom"]):
            if chunk["event"] in ("answer", "image") or chunk["data"].get("step") == "image":
                seen.append(chunk["event"] if chunk["event"] != "step" else "image-start")
        return seen

    assert asyncio.run(events()) == ["answer", "image-start", "image"]  # sequential, never concurrent
    response = run_graph(settings, index, image=FakeImages(runs_after_answer=True), generate_image=True)
    assert response["images"][0]["url"] == "/generated/x.png" and response["answer"]
