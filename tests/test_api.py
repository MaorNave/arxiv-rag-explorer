import json
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from rag_app.server import create_app
from rag_app.service import RAGService

from .conftest import RECORDS, fake_llm, write_jsonl


def wait_ready(client, timeout=30.0, expect_documents=None):
    deadline = time.time() + timeout
    status = None
    while time.time() < deadline:
        status = client.get("/api/status").json()
        documents = (status.get("index") or {}).get("documents")
        if status["ready"] and (expect_documents is None or documents == expect_documents):
            return status
        if status["state"] == "error":
            raise AssertionError(status)
        time.sleep(0.1)
    raise AssertionError(f"not ready: {status}")


@pytest.fixture
def client(settings, embeddings):
    service = RAGService(settings, embeddings=embeddings, llm=fake_llm(), use_ollama=False)
    with TestClient(create_app(settings, service)) as c:
        wait_ready(c)
        yield c


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        name, data = None, None
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:])
        if name:
            events.append((name, data))
    return events


def test_status_and_health(client):
    status = client.get("/api/status").json()
    assert status["state"] == "ready"
    assert status["index"]["documents"] == 5
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/").status_code == 200
    assert "arXiv RAG Explorer" in client.get("/").text


def test_answer_post_and_get(client):
    body = client.post("/answer", json={"query": "What is RLBFF?", "top_k": 2}).json()
    assert {"answer", "citations", "retrieved_context"} <= set(body)
    assert body["citations"][0] == {
        "ref": 1, "doc_id": "2509.00001v1", "title": RECORDS[0]["title"], "url": "https://arxiv.org/abs/2509.00001v1",
    }
    assert len(body["retrieved_context"]) == 2

    get = client.get("/answer", params={"query": "diffusion distillation"})
    assert get.status_code == 200 and get.json()["answer"]

    alias = client.post("/answer", json={"question": "What is RLBFF?"})  # alias accepted
    assert alias.status_code == 200


def test_answer_validation(client):
    assert client.post("/answer", json={"query": ""}).status_code == 422
    assert client.post("/answer", json={"query": "x", "top_k": 0}).status_code == 422


def test_stream_emits_sse_events(client):
    response = client.post("/stream", json={"query": "What is RLBFF?"})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    for expected in ("step", "analysis", "retrieval", "token", "answer", "final", "done"):
        assert expected in names
    final = dict(events)["final"]
    assert final["answer"] and final["citations"] and final["retrieved_context"]

    get = client.get("/stream", params={"query": "hallucination detection"})
    assert "event: final" in get.text


def test_image_request_when_provider_disabled(client):
    body = client.post("/answer", json={"query": "What is RLBFF?", "generate_image": True}).json()
    assert body["images"] == []
    assert "disabled" in body["image_error"]
    assert body["answer"]


def test_upload_new_dataset_rebuilds_index(client, tmp_path):
    path = write_jsonl(tmp_path / "new.jsonl", RECORDS[:2])
    with path.open("rb") as fh:
        response = client.post("/api/dataset", files={"file": ("new.jsonl", fh, "application/json")})
    assert response.status_code == 202, response.text
    status = wait_ready(client, expect_documents=2)
    assert status["index"]["dataset"] == "new.jsonl"

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"title": "no abstract"}\n')
    with bad.open("rb") as fh:
        assert client.post("/api/dataset", files={"file": ("bad.jsonl", fh)}).status_code == 400


def test_service_rejects_queries_before_the_index_exists(settings, embeddings):
    service = RAGService(settings, embeddings=embeddings, llm=fake_llm(), use_ollama=False)
    app = create_app(settings, service)
    client = TestClient(app)  # lifespan not started -> nothing indexed
    assert client.post("/answer", json={"query": "anything"}).status_code == 503
    assert client.post("/stream", json={"query": "anything"}).status_code == 503


def test_service_discovers_the_local_image_server(settings, embeddings, monkeypatch):
    import asyncio

    from rag_app import service as service_module
    IMAGEGEN_URL = "http://127.0.0.1:11435"  # LOCAL_IMAGE_PORT default

    servers = {IMAGEGEN_URL: {"x/flux2-klein:4b"}, settings.ollama_base_url: {"qwen3.5:4b"}}

    def fake_local_models(url):
        if url not in servers:
            raise ConnectionError(url)
        return servers[url]

    monkeypatch.setattr(service_module, "local_models", fake_local_models)
    service = RAGService(replace(settings, image_provider="auto"), embeddings=embeddings, llm=fake_llm(), use_ollama=False)
    asyncio.run(service._refresh_local_image())
    assert service.image_generator.local_url == IMAGEGEN_URL
    assert service.image_generator.runs_after_answer

    del servers[IMAGEGEN_URL]  # the dedicated server was stopped
    asyncio.run(service._refresh_local_image())
    assert service.image_generator.local_url is None
    assert service.image_generator.chain() == ["pollinations"]

    # The regular Ollama has the model too, but 0.32.13 can no longer run it (image generation ended
    # in 0.32.6): where the app runs its own 0.32.5 image server it is skipped, elsewhere it is tried.
    servers[settings.ollama_base_url].add("x/flux2-klein:4b")
    service.ollama_version = "0.32.13"
    monkeypatch.setattr(service, "_local_images_wanted", lambda: True)
    asyncio.run(service._refresh_local_image())
    assert service.image_generator.local_url is None
    monkeypatch.setattr(service, "_local_images_wanted", lambda: False)  # e.g. Linux: the server decides
    asyncio.run(service._refresh_local_image())
    assert service.image_generator.local_url == settings.ollama_base_url


def test_failed_ollama_setup_keeps_its_reason_on_screen(settings, embeddings, monkeypatch):
    import asyncio

    from rag_app import service as service_module

    monkeypatch.setattr(service_module, "ollama_version", lambda url: None)  # no Ollama running
    service = RAGService(settings, embeddings=embeddings, llm=fake_llm())
    monkeypatch.setattr(service.runtime, "can_run_main", lambda: True)

    def offline(models, progress):
        raise RuntimeError("cannot reach github.com")

    monkeypatch.setattr(service.runtime, "start_main", offline)

    async def first_status():
        waiting = asyncio.create_task(service._wait_for_ollama())  # it keeps waiting for Ollama
        for _ in range(500):
            if service.status.snapshot()["state"] == "error":
                break
            await asyncio.sleep(0.01)
        waiting.cancel()

    asyncio.run(first_status())
    status = service.status.snapshot()
    assert status["state"] == "error" and status["error"] == "cannot reach github.com"
    assert "cannot reach github.com" in status["message"]
    assert "OLLAMA_BASE_URL" not in status["message"]  # not the generic advice that just failed


def test_app_is_ready_while_the_reranker_is_still_downloading(settings, embeddings):
    import threading

    release = threading.Event()

    class SlowReranker:
        model = "slow-cross-encoder"
        ready = False

        def load(self):
            release.wait(10)  # e.g. a slow first download from Hugging Face
            self.ready = True

        def score(self, query, texts):
            return [0.0] * len(texts)

    reranker = SlowReranker()
    service = RAGService(settings, embeddings=embeddings, llm=fake_llm(), reranker=reranker, use_ollama=False)
    with TestClient(create_app(settings, service)) as client:
        status = wait_ready(client)
        assert status["models"]["reranker_state"] == "loading"
        body = client.post("/answer", json={"query": "What is RLBFF?"}).json()
        assert body["answer"] and "cross-encoder" not in body["metrics"]["retrieval_mode"]
        release.set()
        deadline = time.time() + 5
        while time.time() < deadline and not reranker.ready:
            time.sleep(0.05)
        body = client.post("/answer", json={"query": "What is RLBFF?"}).json()
        assert "cross-encoder rerank (slow-cross-encoder)" in body["metrics"]["retrieval_mode"]
        assert client.get("/api/status").json()["models"]["reranker_state"] == "ready"


def test_reranker_failure_is_reported_and_answers_keep_working(settings, embeddings):
    class BrokenReranker:
        model = "broken-cross-encoder"
        ready = False

        def load(self):
            raise OSError("Hugging Face is unreachable")

    service = RAGService(settings, embeddings=embeddings, llm=fake_llm(), reranker=BrokenReranker(), use_ollama=False)
    with TestClient(create_app(settings, service)) as client:
        wait_ready(client)
        deadline = time.time() + 5
        while time.time() < deadline and client.get("/api/status").json()["models"]["reranker_state"] != "error":
            time.sleep(0.05)
        models = client.get("/api/status").json()["models"]
        assert models["reranker_state"] == "error" and "unreachable" in models["reranker_error"]
        assert client.post("/answer", json={"query": "What is RLBFF?"}).json()["answer"]
