"""Application service: startup orchestration, dataset watching and query execution."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
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
from .runtime import (
    RuntimeManager,
    import_model,
    is_local_url,
    supports_image_generation,
    system_models_dir,
    total_memory_gb,
)
from .setup_models import (
    INSTALL_HINT,
    ensure_models,
    local_models,
    model_capabilities,
    normalize,
    ollama_version,
    pull_model,
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
        self.runtime = RuntimeManager(settings)
        self.image_setup: dict[str, Any] = {"state": "idle", "message": None, "percent": None}
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
        await asyncio.to_thread(self.runtime.stop)  # the app's own Ollama servers stop with the app

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
            self._build_pipeline(supports_thinking)
            # Optional: the app answers right away and turns reranking on once the model is loaded.
            self._spawn(self._load_reranker())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("Startup failed")
            self.status.set_state("error", f"Startup failed: {exc}", error=str(exc))
            return

        await self.reindex()
        if self.settings.watch_dataset:
            self._spawn(self._watch_dataset())
        if self.use_ollama and self.image_generator.mode in ("auto", "ollama"):
            self._spawn(self._setup_local_images())
            self._spawn(self._watch_local_image())
        if self.use_ollama and self.index.active is not None:
            self._spawn(self._warm_up(supports_thinking))

    async def _wait_for_ollama(self) -> None:
        url = self.settings.ollama_base_url
        self.ollama_version = await asyncio.to_thread(ollama_version, url)
        if not self.ollama_version and self.runtime.can_run_main():
            await self._start_managed_ollama()
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
                f"Waiting for Ollama at {url}. Start it there (or use a local OLLAMA_BASE_URL so the app can "
                "run its own copy); this page updates automatically.",
            )
            await asyncio.sleep(3)

    async def _start_managed_ollama(self) -> None:
        """No Ollama is running: download a private copy into STORAGE_DIR/runtime and run it."""
        s = self.settings

        def progress(percent: float | None, text: str) -> None:
            self.status.update(state="installing_runtime", message=f"No Ollama running: {text}",
                               progress={"percent": percent, "status": text})

        self.status.set_state("installing_runtime", "No Ollama running: setting up the app's own copy…")
        reuse = [s.llm_model, s.embed_model] + ([s.ollama_image_model] if self._local_images_wanted() else [])
        try:
            await asyncio.to_thread(self.runtime.start_main, reuse, progress)
        except Exception as exc:
            log.warning("Could not run the app's own Ollama: %s", exc)
            self.status.set_state("error", f"Could not start Ollama automatically: {exc}", error=str(exc))
            return
        self.ollama_version = await asyncio.to_thread(ollama_version, s.ollama_base_url)

    def _local_images_wanted(self) -> bool:
        s = self.settings
        if s.local_images == "off" or not self.use_ollama or self.image_generator.mode not in ("auto", "ollama"):
            return False
        if sys.platform != "darwin":
            return False  # Ollama's image generation was macOS-first; other systems use the free APIs
        if s.ollama_image_base_url:
            return False  # the user points at their own image server
        if s.local_images == "auto":
            memory = total_memory_gb()
            return memory is not None and memory >= 15.5  # FLUX.2 klein peaks at ~7.6 GB
        return True

    async def _setup_local_images(self) -> None:
        """Make local FLUX images work without any manual step (runs in the background)."""
        if not self._local_images_wanted():
            return
        s = self.settings
        model = normalize(s.ollama_image_model)

        def progress(percent: float | None, text: str) -> None:
            self.image_setup = {"state": "installing", "message": text, "percent": percent}

        try:
            if self.runtime.main is not None:  # the app's own 0.32.5 serves everything, images included
                url = self.runtime.main.url
            elif supports_image_generation(self.ollama_version):  # the regular Ollama can still do it
                url = s.ollama_base_url
            else:  # the regular Ollama is newer than 0.32.5: run the 0.32.5 copy next to it
                progress(None, "setting up the local image server")
                server = await asyncio.to_thread(self.runtime.start_image_server, progress)
                url = server.url
            if model not in await asyncio.to_thread(local_models, url):
                if is_local_url(url) and self.runtime.main is not None:
                    await asyncio.to_thread(import_model, model, system_models_dir(), self.runtime.models_dir)
                if model not in await asyncio.to_thread(local_models, url):
                    await asyncio.to_thread(
                        pull_model, url, model,
                        lambda _m, pct, text: progress(pct, f"downloading {model}: {text}"),
                    )
            self.image_setup = {"state": "ready", "message": f"{model} at {url}", "percent": None}
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Local image setup failed (%s); the free image APIs are used instead", exc)
            self.image_setup = {"state": "error", "message": str(exc), "percent": None}
        await self._refresh_local_image()

    async def _ensure_models(self) -> None:
        s = self.settings
        self.status.set_state("pulling_models", f"Checking local models ({s.llm_model}, {s.embed_model})…")

        def on_progress(model: str, percent: float | None, text: str) -> None:
            label = f"Downloading {model}: {text}" + (f" ({percent:.0f}%)" if percent is not None else "")
            self.status.update(state="pulling_models", message=label,
                               progress={"model": model, "percent": percent, "status": text})

        pulled = await asyncio.to_thread(
            ensure_models, s.ollama_base_url, [s.llm_model, s.embed_model],
            pull=s.auto_pull_models, on_progress=on_progress,
        )
        if pulled:
            log.info("Downloaded Ollama models: %s", ", ".join(pulled))
        await self._refresh_local_image()

    async def _refresh_local_image(self) -> None:
        """Point the image generator at the first Ollama server that has the image model."""
        generator = self.image_generator
        model = normalize(self.settings.ollama_image_model)
        found = None
        for url in generator.local_candidates():
            try:
                if model in await asyncio.to_thread(local_models, url):
                    found = url
                    break
            except Exception:  # server not running
                continue
        if found != generator.local_url:
            if found:
                log.info("Local image model %s available at %s; it is tried first", model, found)
            elif generator.local_url:
                log.info("Local image server %s is gone; using the free image APIs", generator.local_url)
            generator.local_url = found

    async def _watch_local_image(self) -> None:
        """Keep the image provider in sync with which local image servers are running."""
        while True:
            await asyncio.sleep(30)
            await self._refresh_local_image()

    async def _load_reranker(self) -> None:
        """Load (first run: download ~90 MB) the cross-encoder without blocking the app.

        A daemon thread is used so that a slow download can never delay shutting the app down.
        """
        if self.reranker is None or self.reranker.ready:
            return
        loop = asyncio.get_running_loop()
        done: asyncio.Future = loop.create_future()

        def finish(error: BaseException | None = None) -> None:  # runs on the event loop
            if not done.done():
                done.set_result(None) if error is None else done.set_exception(error)

        def work() -> None:
            try:
                self.reranker.load()
            except Exception as exc:  # reported below
                loop.call_soon_threadsafe(finish, exc)
            else:
                loop.call_soon_threadsafe(finish)

        threading.Thread(target=work, name="reranker-load", daemon=True).start()
        try:
            await done
            log.info("Reranking is on")
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
                "reranker_state": (
                    "off" if self.reranker is None else "ready" if self.reranker.ready
                    else "error" if self.reranker_error else "loading"
                ),
                "reranker_error": self.reranker_error,
                "thinking": self.settings.llm_thinking,
                "query_rewrite": self.settings.query_rewrite,
            },
            "ollama": {"base_url": self.settings.ollama_base_url, "version": self.ollama_version},
            "image": {**self.image_generator.describe(), "setup": self.image_setup},
            "runtime": {
                "managed_ollama": self.runtime.main.url if self.runtime.main else None,
                "managed_image_server": self.runtime.images.url if self.runtime.images else None,
            },
            "defaults": {"top_k": self.settings.top_k, "max_top_k": 20},
            "storage_dir": os.fspath(self.settings.storage_dir),
        }

    def example_titles(self, n: int = 4) -> list[str]:
        active = self.index.active
        return active.keywords.random_titles(n) if active is not None else []
