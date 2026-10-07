import json
import time

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
