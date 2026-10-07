"""FastAPI application: web UI + REST (/answer) + Server-Sent Events (/stream)."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import STATIC_DIR, Settings
from .dataset import DatasetError, validate_jsonl_sample
from .index_manager import IndexNotReadyError
from .schemas import AnswerResponse, QueryRequest
from .service import RAGService

log = logging.getLogger(__name__)

API_DESCRIPTION = """\
Retrieval-augmented question answering over a collection of research abstracts.
Everything (indexing, retrieval, generation) runs locally through Ollama; only the
optional image generation calls an external API.

* **POST /answer**: full JSON answer with citations and retrieved context.
* **POST /stream**: the same pipeline as Server-Sent Events (tokens as they are generated).
* GET variants accept `?query=...&top_k=5&generate_image=false` for quick testing.
"""


def _friendly_error(exc: Exception) -> tuple[int, str]:
    if isinstance(exc, IndexNotReadyError):
        return 503, str(exc)
    if isinstance(exc, (httpx.ConnectError, ConnectionError)):
        return 502, "Cannot reach the local Ollama server. Is it running? (ollama serve)"
    name = type(exc).__name__
    if name == "ResponseError":  # ollama.ResponseError, e.g. model not found / out of memory
        return 502, f"Ollama error: {exc}"
    return 500, f"{name}: {exc}"


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(settings: Settings | None = None, service: RAGService | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    service = service or RAGService(settings)
    settings.ensure_dirs()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await service.start()
        try:
            yield
        finally:
            await service.stop()

    app = FastAPI(
        title="arXiv RAG Explorer",
        description=API_DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )
    app.state.service = service

    # ------------------------------------------------------------------ RAG endpoints
    async def run_answer(req: QueryRequest) -> dict:
        try:
            return await service.answer(req.query, top_k=req.top_k, generate_image=req.generate_image)
        except Exception as exc:
            status, message = _friendly_error(exc)
            if status == 500:
                log.exception("Answer failed")
            raise HTTPException(status_code=status, detail=message) from exc

    @app.post("/answer", response_model=AnswerResponse, tags=["RAG"], summary="Answer a question (JSON)")
    async def answer_post(req: QueryRequest):
        return await run_answer(req)

    @app.get("/answer", response_model=AnswerResponse, tags=["RAG"], summary="Answer a question (JSON, GET)")
    async def answer_get(
        query: str = Query(..., min_length=1, max_length=2000),
        top_k: int | None = Query(None, ge=1, le=20),
        generate_image: bool = False,
    ):
        return await run_answer(QueryRequest(query=query, top_k=top_k, generate_image=generate_image))

    def stream_response(req: QueryRequest) -> StreamingResponse:
        try:
            service.ensure_ready()
        except IndexNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        async def events() -> AsyncIterator[str]:
            yield ": stream open\n\n"  # flush headers immediately
            try:
                async for event, data in service.stream(
                    req.query, top_k=req.top_k, generate_image=req.generate_image
                ):
                    yield _sse(event, data)
            except Exception as exc:
                status, message = _friendly_error(exc)
                if status == 500:
                    log.exception("Stream failed")
                yield _sse("error", {"status": status, "message": message})
            yield _sse("done", {})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    stream_doc = (
        "Server-Sent Events. Event types: `step`, `analysis`, `retrieval`, `token`, `thinking`, "
        "`answer` (text complete), `image`, `final` (full JSON, same shape as /answer), `error`, `done`."
    )

    @app.post("/stream", tags=["RAG"], summary="Answer a question (streaming SSE)", description=stream_doc,
              response_class=StreamingResponse)
    async def stream_post(req: QueryRequest):
        return stream_response(req)

    @app.get("/stream", tags=["RAG"], summary="Answer a question (streaming SSE, GET / EventSource)",
             description=stream_doc, response_class=StreamingResponse)
    async def stream_get(
        query: str = Query(..., min_length=1, max_length=2000),
        top_k: int | None = Query(None, ge=1, le=20),
        generate_image: bool = False,
    ):
        return stream_response(QueryRequest(query=query, top_k=top_k, generate_image=generate_image))

    # ------------------------------------------------------------------ operations
    @app.get("/api/status", tags=["System"], summary="Startup / indexing status, models and dataset info")
    async def status():
        return service.status_payload()

    @app.get("/health", tags=["System"], summary="Liveness/readiness probe")
    async def health():
        payload = service.status_payload()
        code = 200 if payload["ready"] else 503
        return JSONResponse({"status": "ok" if payload["ready"] else payload["state"]}, status_code=code)

    @app.get("/api/examples", tags=["System"], summary="Random paper titles to inspire questions")
    async def examples(n: int = Query(4, ge=1, le=12)):
        return {"titles": service.example_titles(n)}

    @app.post("/api/reindex", tags=["System"], summary="Discard the index and rebuild it from DATA_PATH", status_code=202)
    async def reindex():
        if not service.schedule_reindex(force=True):
            raise HTTPException(status_code=409, detail="Indexing is already in progress")
        return {"status": "indexing"}

    @app.post("/api/dataset", tags=["System"], summary="Upload a new .jsonl dataset (replaces the index)",
              status_code=202)
    async def upload_dataset(file: UploadFile = File(...)):
        name = Path(file.filename or "dataset.jsonl").name
        if not name.lower().endswith(".jsonl"):
            raise HTTPException(status_code=400, detail="Please upload a .jsonl file")
        if service.is_indexing:
            raise HTTPException(status_code=409, detail="Indexing is already in progress")
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        target = settings.uploads_dir / safe_name
        partial = target.with_suffix(".part")
        limit = settings.max_upload_mb * 1024 * 1024
        size = 0
        try:
            with partial.open("wb") as fh:
                while block := await file.read(1 << 20):
                    size += len(block)
                    if size > limit:
                        raise HTTPException(status_code=413, detail=f"File is larger than {settings.max_upload_mb} MB")
                    fh.write(block)
            validate_jsonl_sample(partial)
        except DatasetError as exc:
            partial.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        os.replace(partial, target)
        if not service.schedule_reindex(data_path=str(target)):
            raise HTTPException(status_code=409, detail="Indexing is already in progress")
        return {"status": "indexing", "dataset": safe_name, "bytes": size}

    # ------------------------------------------------------------------ web UI
    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.mount("/generated", StaticFiles(directory=settings.images_dir, check_dir=False), name="generated")
    return app
