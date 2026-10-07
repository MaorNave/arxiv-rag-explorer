"""Application service: startup orchestration, dataset watching and query execution."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from . import __version__
from .config import Settings
from .dataset import DatasetError, file_sha256, is_url, resolve_dataset_path
from .embeddings import PrefixedEmbeddings, build_embeddings
from .graph import PipelineDeps, build_graph, initial_state
from .images import ImageGenerator
from .index_manager import BuildCancelledError, IndexManager, IndexNotReadyError
from .llm import build_analyzer, build_chat_model
from .reranker import CrossEncoderReranker
from .retriever import HybridRetriever
from .setup_models import (
    INSTALL_HINT,
    ensure_models,
    local_models,
    model_capabilities,
    normalize,
    ollama_version,
)
from .status import StatusTracker

log = logging.getLogger(__name__)


def _signature(path: Path) -> tuple[str, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return str(path), stat.st_size, stat.st_mtime_ns


class RAGService:
    """Owns the index, the LangGraph pipeline and the background lifecycle tasks.

    The constructor accepts pre-built components so tests can run the full stack
    with fake embeddings / chat models and without an Ollama server.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        embeddings: Embeddings | None = None,
        llm: BaseChatModel | None = None,
        analyzer: Runnable | None = None,
        image_generator: ImageGenerator | None = None,
        reranker: CrossEncoderReranker | None = None,
        use_ollama: bool = True,
    ):
        self.settings = settings
        self.status = StatusTracker()
        if embeddings is not None and not isinstance(embeddings, PrefixedEmbeddings):
            embeddings = PrefixedEmbeddings(embeddings, model=settings.embed_model)
        self.embeddings = embeddings or build_embeddings(settings)
        self.index = IndexManager(settings, self.embeddings, self.status)
        self.image_generator = image_generator or ImageGenerator(settings)
        if reranker is None and settings.reranker_enabled:
            reranker = CrossEncoderReranker(settings.reranker_model, settings.models_dir, settings.rerank_max_tokens)
        self.reranker = reranker
        self.reranker_error: str | None = None
        self.use_ollama = use_ollama
        self.data_path = settings.data_path
        self.dataset_path: Path | None = None
        self.ollama_version: str | None = None
        self.graph = None
        self._llm = llm
        self._analyzer = analyzer
        self._tasks: set[asyncio.Task] = set()
        self._reindex_lock = asyncio.Lock()
        self._watch_sig: tuple | None = None

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self.settings.ensure_dirs()
        self._spawn(self._bootstrap())

    async def stop(self) -> None:
        self.index.stop()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _bootstrap(self) -> None:
        try:
            supports_thinking = True
            if self.use_ollama:
                await self._wait_for_ollama()
                await self._ensure_models()
                capabilities = await asyncio.to_thread(
                    model_capabilities, self.settings.ollama_base_url, self.settings.llm_model
                )
                supports_thinking = "thinking" in capabilities if capabilities else True
            await self._load_reranker()
            self._build_pipeline(supports_thinking)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("Startup failed")
            self.status.set_state("error", f"Startup failed: {exc}", error=str(exc))
            return

        await self.reindex()
        if self.settings.watch_dataset:
            self._spawn(self._watch_dataset())
        if self.use_ollama and self.index.active is not None:
            self._spawn(self._warm_up(supports_thinking))

    async def _wait_for_ollama(self) -> None:
        url = self.settings.ollama_base_url
        hinted = False
        while True:
            self.ollama_version = await asyncio.to_thread(ollama_version, url)
            if self.ollama_version:
                log.info("Connected to Ollama %s at %s", self.ollama_version, url)
                return
            if not hinted:
                log.warning(INSTALL_HINT.format(url=url))
                hinted = True
            self.status.set_state(
                "waiting_for_ollama",
                f"Waiting for Ollama at {url}. Install it from ollama.com and start it; this page updates automatically.",
            )
            await asyncio.sleep(3)

    async def _ensure_models(self) -> None:
        s = self.settings
        self.status.set_state("pulling_models", f"Checking local models ({s.llm_model}, {s.embed_model})…")

        def on_progress(model: str, percent: float | None, text: str) -> None:
            label = f"Downloading {model}: {text}" + (f" ({percent:.0f}%)" if percent is not None else "")
            self.status.update(state="pulling_models", message=label,
                               progress={"model": model, "percent": percent, "status": text})

        models = [s.llm_model, s.embed_model]
        if s.image_provider == "ollama":  # explicitly chosen local image model: download it too
            models.append(s.ollama_image_model)
        pulled = await asyncio.to_thread(
            ensure_models, s.ollama_base_url, models,
            pull=s.auto_pull_models, on_progress=on_progress,
        )
        if pulled:
            log.info("Downloaded Ollama models: %s", ", ".join(pulled))
        installed = await asyncio.to_thread(local_models, s.ollama_base_url)
        self.image_generator.local_installed = normalize(s.ollama_image_model) in installed
        if self.image_generator.local_installed:
            log.info("Local image model %s found; it is tried first for illustrations", s.ollama_image_model)

    async def _load_reranker(self) -> None:
        if self.reranker is None or self.reranker.ready:
            return
        self.status.set_state(
            "pulling_models", f"Loading the reranker {self.reranker.model} (first run downloads ~90 MB)…"
        )
        try:
            await asyncio.to_thread(self.reranker.load)
        except Exception as exc:  # the pipeline still works, just without reranking
            self.reranker_error = f"{type(exc).__name__}: {exc}"
            log.warning("Reranker unavailable (%s); continuing with hybrid ranking only", self.reranker_error)

    def _build_pipeline(self, supports_thinking: bool) -> None:
        llm = self._llm or build_chat_model(self.settings, supports_thinking=supports_thinking)
        analyzer = self._analyzer
        if analyzer is None and self.settings.query_rewrite and self._llm is None:
            analyzer = build_analyzer(self.settings, supports_thinking=supports_thinking)
        deps = PipelineDeps(
            settings=self.settings,
            llm=llm,
            get_retriever=self.get_retriever,
            analyzer=analyzer,
            image_generator=self.image_generator,
            reranker=self.reranker,
        )
        self.graph = build_graph(deps)

    async def _warm_up(self, supports_thinking: bool) -> None:
        """Load both models into memory so the first real question is fast."""
        try:
            # A throwaway search loads the embedding model and Chroma's HNSW index into memory.
            await asyncio.to_thread(self.get_retriever().search, ["warm up"], [], 1)
            if self.reranker is not None and self.reranker.ready:
                await asyncio.to_thread(self.reranker.score, "warm up", ["warm up"])
            warm = build_chat_model(self.settings, num_predict=1, thinking=False, supports_thinking=supports_thinking)
            await warm.ainvoke("Reply with OK.")
            log.info("Models warmed up")
        except Exception as exc:
            log.warning("Model warm-up failed: %s", exc)

    # ------------------------------------------------------------------ indexing
    @property
    def is_indexing(self) -> bool:
        return self._reindex_lock.locked()

    async def reindex(self, *, force: bool = False, data_path: str | None = None) -> None:
        async with self._reindex_lock:
            if data_path is not None:
                self.data_path = data_path
            try:
                path = await asyncio.to_thread(resolve_dataset_path, self.data_path, self.settings.downloads_dir)
                self.dataset_path = path
                self._watch_sig = _signature(path)
                await asyncio.to_thread(self.index.load_or_build, path, force=force)
                manifest = self.index.active.manifest
                self.status.set_state(
                    "ready",
                    f"Ready: {manifest['dataset']['documents']:,} abstracts indexed from {manifest['dataset']['name']}",
                )
            except (asyncio.CancelledError, BuildCancelledError):
                raise
            except Exception as exc:
                log.exception("Indexing failed")
                self.status.set_state("error", f"Indexing failed: {exc}", error=str(exc))

    def schedule_reindex(self, *, force: bool = False, data_path: str | None = None) -> bool:
        if self.is_indexing:
            return False
        self._spawn(self.reindex(force=force, data_path=data_path))
        return True

    async def _watch_dataset(self) -> None:
        """Rebuild automatically when the dataset file is replaced while the server runs."""
        pending: tuple | None = None
        while True:
            await asyncio.sleep(self.settings.watch_interval_s)
            if self.is_indexing or is_url(self.data_path):
                continue
            try:
                path = await asyncio.to_thread(resolve_dataset_path, self.data_path, None)
            except DatasetError:
                continue
            sig = _signature(path)
            if sig is None or sig == self._watch_sig:
                pending = None
                continue
            if sig != pending:  # wait one more tick so half-copied files are not indexed
                pending = sig
                continue
            pending = None
            active = self.index.active
            if (
                active is not None
                and path == self.dataset_path
                and await asyncio.to_thread(file_sha256, path) == active.dataset_sha256
            ):
                self._watch_sig = sig  # touched but unchanged
                continue
            log.info("Dataset change detected (%s): discarding the old index and rebuilding", path.name)
            await self.reindex()

    # ------------------------------------------------------------------ queries
    def get_retriever(self) -> HybridRetriever:
        return HybridRetriever(index=self.index.require_active(), k=self.settings.top_k, fetch_k=self.settings.fetch_k)

    def ensure_ready(self) -> None:
        if self.graph is None or self.index.active is None:
            raise IndexNotReadyError(self.status.snapshot().get("message") or "The service is starting")

    def _inputs(self, question: str, top_k: int | None, generate_image: bool) -> dict[str, Any]:
        return initial_state(question, top_k or self.settings.top_k, generate_image)

    async def answer(self, question: str, *, top_k: int | None = None, generate_image: bool = False) -> dict:
        self.ensure_ready()
        state = await self.graph.ainvoke(self._inputs(question, top_k, generate_image))
        return state["response"]

    async def stream(
        self, question: str, *, top_k: int | None = None, generate_image: bool = False
    ) -> AsyncIterator[tuple[str, dict]]:
        """Yields (event, data) pairs: step, analysis, retrieval, token, thinking, answer, image, final."""
        self.ensure_ready()
        inputs = self._inputs(question, top_k, generate_image)
        async for mode, chunk in self.graph.astream(inputs, stream_mode=["custom", "updates"]):
            if mode == "custom":
                yield chunk["event"], chunk["data"]
            elif mode == "updates" and "finalize" in chunk:
                yield "final", chunk["finalize"]["response"]

    # ------------------------------------------------------------------ info
    def status_payload(self) -> dict[str, Any]:
        snap = self.status.snapshot()
        active = self.index.active
        index_info = None
        if active is not None:
            m = active.manifest
            index_info = {
                "dataset": m["dataset"]["name"],
                "dataset_path": m["dataset"]["path"],
                "sha256": m["dataset"]["sha256"],
                "size_bytes": m["dataset"]["size_bytes"],
                "documents": m["dataset"]["documents"],
                "duplicates": m["dataset"]["duplicates"],
                "malformed": m["dataset"]["malformed"],
                "missing_text": m["dataset"].get("missing_text", 0),
                "chunks": m["chunks"],
                "embedding_dimensions": m["embedding"]["dimensions"],
                "built_at": m["built_at"],
                "build_seconds": m["build_seconds"],
                "fingerprint": m["fingerprint"][:12],
                "vector_store": "Chroma (persistent, cosine)",
                "keyword_index": "SQLite FTS5 (BM25)",
            }
        return {
            "ready": snap["state"] == "ready" and active is not None and self.graph is not None,
            "state": snap["state"],
            "message": snap["message"],
            "progress": snap["progress"],
            "error": snap["error"],
            "version": __version__,
            "data_path": self.data_path,
            "index": index_info,
            "models": {
                "llm": self.settings.llm_model,
                "embedding": self.settings.embed_model,
                "reranker": self.reranker.model if self.reranker is not None and self.reranker.ready else None,
                "reranker_error": self.reranker_error,
                "thinking": self.settings.llm_thinking,
                "query_rewrite": self.settings.query_rewrite,
            },
            "ollama": {"base_url": self.settings.ollama_base_url, "version": self.ollama_version},
            "image": self.image_generator.describe(),
            "defaults": {"top_k": self.settings.top_k, "max_top_k": 20},
            "storage_dir": os.fspath(self.settings.storage_dir),
        }

    def example_titles(self, n: int = 4) -> list[str]:
        active = self.index.active
        return active.keywords.random_titles(n) if active is not None else []
