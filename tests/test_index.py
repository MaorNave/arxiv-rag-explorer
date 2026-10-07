from dataclasses import replace

from rag_app.embeddings import PrefixedEmbeddings
from rag_app.index_manager import IndexManager
from rag_app.keyword_index import build_match_query
from rag_app.retriever import HybridRetriever

from .conftest import RECORDS, write_jsonl


def make_manager(settings, embeddings):
    return IndexManager(settings, PrefixedEmbeddings(embeddings, model="fake"))


def test_build_then_reuse_without_rebuilding(settings, embeddings, dataset):
    manager = make_manager(settings, embeddings)
    index = manager.load_or_build(dataset)
    ds = index.manifest["dataset"]
    assert (ds["documents"], ds["duplicates"], ds["malformed"], ds["missing_text"]) == (5, 1, 1, 1)
    assert index.manifest["chunks"] == 5
    assert index.keywords.count_chunks() == 5

    calls = embeddings.calls
    # A fresh manager (= app restart) must load the persisted index, not re-embed.
    reloaded = make_manager(settings, embeddings).load_or_build(dataset)
    assert reloaded.manifest["collection"] == index.manifest["collection"]
    assert embeddings.calls == calls


def test_new_dataset_discards_old_index_and_rebuilds(settings, embeddings, dataset):
    manager = make_manager(settings, embeddings)
    old = manager.load_or_build(dataset)
    old_collection = old.manifest["collection"]

    write_jsonl(dataset, RECORDS[:2])  # dataset replaced
    new = manager.load_or_build(dataset)
    assert new.manifest["collection"] != old_collection
    assert new.manifest["dataset"]["documents"] == 2
    names = {c if isinstance(c, str) else c.name for c in manager.client.list_collections()}
    assert names == {new.manifest["collection"]}  # the old vectors are gone
    assert len(list(settings.index_dir.glob("keywords-*.sqlite"))) == 1


def test_changed_index_settings_trigger_rebuild(settings, embeddings, dataset):
    first = make_manager(settings, embeddings).load_or_build(dataset)
    smaller_chunks = replace(settings, chunk_size=60, chunk_overlap=10)
    second = make_manager(smaller_chunks, embeddings).load_or_build(dataset)
    assert second.manifest["fingerprint"] != first.manifest["fingerprint"]
    assert second.manifest["chunks"] > second.manifest["dataset"]["documents"]  # long abstracts were split


def test_forced_rebuild(settings, embeddings, dataset):
    manager = make_manager(settings, embeddings)
    first = manager.load_or_build(dataset)
    second = manager.load_or_build(dataset, force=True)
    assert second.manifest["collection"] != first.manifest["collection"]


def test_keyword_match_query_is_safe():
    query = build_match_query(['What is "RLBFF" OR NEAR(x)?'], phrases=["reward models"])
    assert query.startswith('"reward models"')
    assert '"rlbff"' in query and '"near"' in query
    assert build_match_query(["the of and"]) is None  # stopwords only


def test_hybrid_retrieval_finds_relevant_papers(settings, embeddings, dataset):
    index = make_manager(settings, embeddings).load_or_build(dataset)
    retriever = HybridRetriever(index=index, k=3, fetch_k=5)

    results = retriever.search(["What is RLBFF?"])
    assert results[0].doc_id == "2509.00001v1"
    assert results[0].keyword_rank == 1
    assert results[0].similarity is not None
    assert len({p.doc_id for p in results}) == len(results)  # one passage per paper

    # LangChain retriever interface
    docs = retriever.invoke("hallucinations in language models")
    assert docs[0].metadata["doc_id"] == "2509.00003v1"

    # Ablation modes
    dense = HybridRetriever(index=index, k=3, fetch_k=5, mode="dense").search(["gaussian splatting avatars"])
    assert dense[0].doc_id == "2509.00004v1" and dense[0].keyword_rank is None
    sparse = HybridRetriever(index=index, k=3, fetch_k=5, mode="keyword").search(["diffusion distillation"])
    assert sparse[0].doc_id == "2509.00002v1" and sparse[0].dense_rank is None


def test_rewrite_weights_cannot_outvote_the_question(settings, embeddings, dataset):
    index = make_manager(settings, embeddings).load_or_build(dataset)
    retriever = HybridRetriever(index=index, k=3, fetch_k=5, keyword_weight=1.5)
    results = retriever.search(
        ["What is RLBFF?", "graph neural networks molecules", "message passing atoms bonds"],
        weights=[1.0, 0.25, 0.25],
    )
    assert results[0].doc_id == "2509.00001v1"
