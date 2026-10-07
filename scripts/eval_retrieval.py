"""Retrieval benchmark: does the right abstract come back?

Three automatically built query sets over the indexed dataset:

* semantic     For N random papers the local LLM writes the question a researcher would
               type into a search engine, without naming the paper or its method.
* paraphrased  Same papers, but the question must avoid the abstract's technical terms
               (vocabulary mismatch, closer to how non-experts ask).
* named        "What is <NAME>?" for papers whose title starts with an acronym/name,
               e.g. "RLBFF: ..." (tests exact-term lookup).

Each query has one relevant paper. We report hit@1, hit@k and MRR@k for:

    dense             vector search (Chroma) with the raw question
    keyword           BM25 (SQLite FTS5) with the raw question
    hybrid            dense + BM25 fused with Reciprocal Rank Fusion
    hybrid+plan       + LLM query plan (rewrites + keywords), weights tuned for no reranker
    hybrid+plan+rerank  the full pipeline: broad recall + cross-encoder reranking

Usage (stop the web app first so both do not compete for the GPU):
    python scripts/eval_retrieval.py --n 40 --k 5
Generated questions and query plans are cached in STORAGE_DIR/eval_*.json
(delete them to regenerate).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_app.config import Settings  # noqa: E402
from rag_app.dataset import resolve_dataset_path  # noqa: E402
from rag_app.embeddings import build_embeddings  # noqa: E402
from rag_app.graph import _mostly_latin, fusion_weights, retrieval_inputs  # noqa: E402
from rag_app.index_manager import IndexManager  # noqa: E402
from rag_app.llm import build_analyzer, build_chat_model  # noqa: E402
from rag_app.prompts import ANALYSIS_PROMPT  # noqa: E402
from rag_app.reranker import CrossEncoderReranker, blend_rankings  # noqa: E402
from rag_app.retriever import HybridRetriever  # noqa: E402

QUESTION_PROMPT = """\
Here is a research paper abstract:

{abstract}

Write ONE question that this abstract answers, phrased the way a researcher would type it \
into a search engine. Do not mention the paper's title, acronyms or the name of the proposed \
method. Output only the question."""

PARAPHRASE_PROMPT = """\
Here is a research paper abstract:

{abstract}

Write ONE question that a curious engineer who has not read this paper might ask, and that \
this abstract answers. Use everyday wording: do NOT reuse the abstract's distinctive technical \
terms, method names, acronyms or dataset names; describe the idea instead. Output only the question."""

NAMED_TITLE = re.compile(r"^([A-Z][A-Za-z0-9\-\.\+$^]{2,24}):\s")


def sample_documents(db: Path, n: int, seed: int) -> tuple[list, list]:
    con = sqlite3.connect(db)
    rows = con.execute("SELECT doc_id, title, body FROM chunks WHERE chunk_id LIKE '%#0'").fetchall()
    con.close()
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows))), rows


def build_queries(settings: Settings, docs, all_rows, n_named: int, seed: int, cache: Path) -> list[dict]:
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    llm = build_chat_model(settings, temperature=0.7, num_predict=80, thinking=False)
    queries = []
    for query_set, prompt, prefix in (("semantic", QUESTION_PROMPT, ""), ("paraphrased", PARAPHRASE_PROMPT, "para:")):
        for doc_id, title, abstract in docs:
            question = cached.get(prefix + doc_id)
            if not question:
                question = llm.invoke(prompt.format(abstract=abstract)).text.strip().strip('"')
                cached[prefix + doc_id] = question
            queries.append({"set": query_set, "question": question, "doc_id": doc_id, "title": title})
    cache.write_text(json.dumps(cached, indent=1))

    named = [(d, t, NAMED_TITLE.match(t).group(1)) for d, t, _ in all_rows if NAMED_TITLE.match(t)]
    random.Random(seed).shuffle(named)
    for doc_id, title, name in named[:n_named]:
        queries.append({"set": "named", "question": f"What is {name}?", "doc_id": doc_id, "title": title})
    return queries


def rank_of(passages, doc_id: str) -> int | None:
    for i, p in enumerate(passages, start=1):
        if p.doc_id == doc_id:
            return i
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=40, help="semantic queries (random papers)")
    parser.add_argument("--named", type=int, default=20, help="'What is <NAME>?' queries")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    settings = Settings.from_env()
    embeddings = build_embeddings(settings)
    manager = IndexManager(settings, embeddings)
    index = manager.load_or_build(resolve_dataset_path(settings.data_path, settings.downloads_dir))
    db = settings.index_dir / index.manifest["keyword_db"]
    docs, all_rows = sample_documents(db, args.n, args.seed)
    print(f"Building queries for {len(docs)} papers (+{args.named} named)…", flush=True)
    queries = build_queries(settings, docs, all_rows, args.named, args.seed, settings.storage_dir / "eval_questions.json")

    analyzer = build_analyzer(settings)
    plan_cache_path = settings.storage_dir / "eval_plans.json"
    plan_cache = json.loads(plan_cache_path.read_text()) if plan_cache_path.exists() else {}
    strategies = ["dense", "keyword", "hybrid", "hybrid+plan"]
    reranker = None
    if settings.reranker_enabled:
        reranker = CrossEncoderReranker(settings.reranker_model, settings.models_dir, settings.rerank_max_tokens)
        reranker.load()
        strategies.append("hybrid+plan+rerank")
    results = {s: {"semantic": [], "paraphrased": [], "named": []} for s in strategies}
    rerank_seconds = []
    plan_seconds = []
    for i, q in enumerate(queries, start=1):
        for strategy in strategies:
            mode = "hybrid" if strategy.startswith("hybrid") else strategy
            reranking = strategy.endswith("rerank")
            rewrite_weight, keyword_weight = fusion_weights(settings, reranking)
            retriever = HybridRetriever(
                index=index, k=args.k, fetch_k=settings.fetch_k, mode=mode,
                keyword_weight=keyword_weight if strategy.startswith("hybrid+plan") else 1.0,
            )
            if strategy.startswith("hybrid+plan"):
                plan_dict = plan_cache.get(q["question"])
                if plan_dict is None:
                    started = time.perf_counter()
                    plan = analyzer.invoke(ANALYSIS_PROMPT.format_messages(question=q["question"]))
                    plan_seconds.append(time.perf_counter() - started)
                    plan_dict = {"search_queries": plan.search_queries, "keywords": plan.keywords}
                    plan_cache[q["question"]] = plan_dict
                queries, weights, keywords = retrieval_inputs(q["question"], plan_dict, rewrite_weight)
                limit = settings.rerank_candidates if reranking else args.k
                passages = retriever.search(queries, keywords, limit, weights=weights)
                if reranking:
                    query = q["question"] if _mostly_latin(q["question"]) else (plan_dict["search_queries"] or [q["question"]])[0]
                    started = time.perf_counter()
                    scores = reranker.score(query, [f"{p.title}. {p.text}" for p in passages])
                    rerank_seconds.append(time.perf_counter() - started)
                    passages = [passages[i] for i, _, _ in blend_rankings(scores, settings.rerank_weight)[: args.k]]
            else:
                passages = retriever.search([q["question"]], [], args.k)
            results[strategy][q["set"]].append(rank_of(passages, q["doc_id"]))
        print(f"  [{i}/{len(queries)}] {q['set']:8s} {q['question'][:90]}", flush=True)
    plan_cache_path.write_text(json.dumps(plan_cache, indent=1))

    def stats(ranks: list[int | None]) -> tuple[float, float, float]:
        n = len(ranks) or 1
        hit1 = sum(1 for r in ranks if r == 1) / n
        hitk = sum(1 for r in ranks if r is not None) / n
        mrr = sum(1 / r for r in ranks if r is not None) / n
        return hit1, hitk, mrr

    print(f"\nLLM: {settings.llm_model} | embeddings: {settings.embed_model} | "
          f"reranker: {settings.reranker_model if reranker else 'off'} | k={args.k}\n")
    for query_set in ("semantic", "paraphrased", "named"):
        count = len(results["dense"][query_set])
        if not count:
            continue
        print(f"{query_set} queries (n={count})\n")
        print(f"| strategy | hit@1 | hit@{args.k} | MRR@{args.k} |")
        print("|---|---|---|---|")
        for strategy in strategies:
            h1, hk, mrr = stats(results[strategy][query_set])
            print(f"| {strategy} | {h1:.2f} | {hk:.2f} | {mrr:.2f} |")
        print()
    if plan_seconds:
        print(f"Query planning latency: median {sorted(plan_seconds)[len(plan_seconds) // 2]:.1f}s")
    if rerank_seconds:
        print(f"Reranking latency: median {1000 * sorted(rerank_seconds)[len(rerank_seconds) // 2]:.0f} ms")


if __name__ == "__main__":
    main()
