"""Builds, loads and swaps the local index.

Layout on disk (``STORAGE_DIR/index``)::

    chroma/                       Chroma persistent client (dense vectors, cosine space)
    keywords-<fp>-<ts>.sqlite     SQLite FTS5 table (BM25 keyword search)
    manifest.json                 describes the active index (written last = commit point)

The *fingerprint* of an index combines the SHA-256 of the dataset file with every
setting that changes the stored vectors (embedding model, prefixes, chunking).
On startup the index is reused only if the fingerprint matches; otherwise the old
index is discarded and a new one is built. A crash mid-build never corrupts the
active index because ``manifest.json`` is replaced atomically only after success.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .dataset import DatasetError, JsonlAbstractLoader, count_records, file_sha256
from .embeddings import PrefixedEmbeddings
from .keyword_index import KeywordIndex
from .status import StatusTracker

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
T = TypeVar("T")


class IndexNotReadyError(RuntimeError):
    """Raised when a query arrives before an index is available."""


class BuildCancelledError(RuntimeError):
    """Raised when the server shuts down while an index is being built."""


@dataclass
class ActiveIndex:
    manifest: dict[str, Any]
    vectorstore: Chroma
    keywords: KeywordIndex
    embeddings: PrefixedEmbeddings

    @property
    def dataset_sha256(self) -> str:
        return self.manifest["dataset"]["sha256"]


class IndexManager:
    def __init__(self, settings: Settings, embeddings: PrefixedEmbeddings, status: StatusTracker | None = None):
        self.settings = settings
        self.embeddings = embeddings
        self.status = status or StatusTracker()
        self.root = settings.index_dir
        self.manifest_path = self.root / "manifest.json"
        self._client: chromadb.ClientAPI | None = None
        self._active: ActiveIndex | None = None
        self._build_lock = threading.Lock()
        self._stop = threading.Event()

    # ------------------------------------------------------------------ public API
    @property
    def active(self) -> ActiveIndex | None:
        return self._active

    @property
    def is_building(self) -> bool:
        return self._build_lock.locked()

    def require_active(self) -> ActiveIndex:
        index = self._active
        if index is None:
            snap = self.status.snapshot()
            raise IndexNotReadyError(snap.get("message") or "The index is not ready yet")
        return index

    def stop(self) -> None:
        """Ask a running build to stop (used on server shutdown)."""
        self._stop.set()

    def fingerprint(self, dataset_sha256: str) -> str:
        payload = {
            "schema": SCHEMA_VERSION,
            "dataset_sha256": dataset_sha256,
            "embed_model": self.settings.embed_model,
            "doc_prefix": self.embeddings.doc_prefix,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def load_or_build(self, dataset_path: Path, *, force: bool = False) -> ActiveIndex:
        """Reuse the persisted index when it matches the dataset, otherwise rebuild it."""
        with self._build_lock:
            self.root.mkdir(parents=True, exist_ok=True)
            self.status.set_state("indexing", f"Fingerprinting {dataset_path.name}…")
            dataset_sha = file_sha256(dataset_path)
            fingerprint = self.fingerprint(dataset_sha)

            if not force:
                current = self._active
                if current is not None and current.manifest["fingerprint"] == fingerprint:
                    return current
                loaded = self._try_load(fingerprint, dataset_path)
                if loaded is not None:
                    log.info(
                        "Loaded existing index (%s documents, %s chunks) for %s",
                        loaded.manifest["dataset"]["documents"], loaded.manifest["chunks"], dataset_path.name,
                    )
                    self._set_active(loaded)
                    self._cleanup()
                    return loaded

            previous = self._read_manifest()
            if force:
                reason = "manual rebuild requested"
            elif previous is None:
                reason = "no existing index"
            elif previous.get("dataset", {}).get("sha256") != dataset_sha:
                reason = "dataset changed"
            else:
                reason = "index settings changed"
            log.info("Building a new index for %s (%s)", dataset_path.name, reason)

            # Discard the old index: stop serving it before the new one is built.
            self._set_active(None)
            built = self._build(dataset_path, dataset_sha, fingerprint)
            self._set_active(built)
            self._cleanup()
            return built

    # ------------------------------------------------------------------ internals
    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.root / "chroma"),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def _set_active(self, index: ActiveIndex | None) -> None:
        old, self._active = self._active, index
        if old is not None and old is not index:
            old.keywords.close()

    def _read_manifest(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp, self.manifest_path)

    def _open(self, manifest: dict[str, Any]) -> ActiveIndex:
        vectorstore = Chroma(
            client=self.client,
            collection_name=manifest["collection"],
            embedding_function=self.embeddings,
            create_collection_if_not_exists=False,
        )
        keywords = KeywordIndex(self.root / manifest["keyword_db"])
        return ActiveIndex(manifest=manifest, vectorstore=vectorstore, keywords=keywords, embeddings=self.embeddings)

    def _try_load(self, fingerprint: str, dataset_path: Path) -> ActiveIndex | None:
        manifest = self._read_manifest()
        if manifest is None or manifest.get("fingerprint") != fingerprint:
            return None
        try:
            vectors = self.client.get_collection(manifest["collection"]).count()
            if not (self.root / manifest["keyword_db"]).exists():
                raise FileNotFoundError(manifest["keyword_db"])
            index = self._open(manifest)
            keywords = index.keywords.count_chunks()
        except Exception as exc:  # missing/corrupt pieces -> rebuild
            log.warning("Existing index is unusable (%s); rebuilding", exc)
            return None
        if vectors != manifest["chunks"] or keywords != manifest["chunks"]:
            log.warning("Index is incomplete (%s vectors, %s keyword rows, expected %s); rebuilding",
                        vectors, keywords, manifest["chunks"])
            index.keywords.close()
            return None
        # Same content under a different file name/location: keep the index, refresh the label.
        if manifest["dataset"].get("path") != str(dataset_path):
            manifest["dataset"].update(path=str(dataset_path), name=dataset_path.name)
            self._write_manifest(manifest)
        return index

    def _with_retries(self, fn: Callable[[], T], attempts: int = 3) -> T:
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:
                if attempt == attempts:
                    raise
                log.warning("Embedding batch failed (%s); retrying (%d/%d)", exc, attempt, attempts - 1)
                time.sleep(2 * attempt)
        raise AssertionError("unreachable")

    def _build(self, dataset_path: Path, dataset_sha: str, fingerprint: str) -> ActiveIndex:
        self._stop.clear()
        started = time.monotonic()
        stamp = f"{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"  # unique per build
        collection_name = f"abstracts-{fingerprint[:12]}-{stamp}"
        keyword_db = f"keywords-{fingerprint[:12]}-{stamp}.sqlite"

        total = count_records(dataset_path)
        progress = {"processed": 0, "total": total, "percent": 0.0, "eta_s": None, "dataset": dataset_path.name}
        self.status.set_state("indexing", f"Indexing {total:,} records from {dataset_path.name}…", progress=progress)

        vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
        keywords = KeywordIndex(self.root / keyword_db)
        keywords.create()

        loader = JsonlAbstractLoader(dataset_path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size, chunk_overlap=self.settings.chunk_overlap
        )
        batch: list[Document] = []
        documents = chunks = duplicates = 0

        def flush() -> None:
            nonlocal chunks
            if not batch:
                return
            if self._stop.is_set():
                raise BuildCancelledError("Index build cancelled")
            ids = [doc.id for doc in batch]
            self._with_retries(lambda: vectorstore.add_documents(batch, ids=ids))
            keywords.add_chunks(
                (doc.id, doc.metadata["doc_id"], doc.metadata["title"], doc.page_content) for doc in batch
            )
            chunks += len(batch)
            batch.clear()
            processed = loader.stats.lines
            elapsed = time.monotonic() - started
            rate = processed / elapsed if elapsed > 0 else 0
            self.status.update(progress={
                **progress,
                "processed": processed,
                "percent": round(100 * processed / max(total, 1), 1),
                "eta_s": round((total - processed) / rate) if rate > 0 else None,
                "docs_per_s": round(rate, 1),
            })

        try:
            for doc in loader.lazy_load():
                doc_id, title = doc.metadata["doc_id"], doc.metadata["title"]
                if not keywords.register_doc(doc_id, title):
                    duplicates += 1
                    continue
                text = doc.page_content
                pieces = [text] if len(text) <= self.settings.chunk_size else splitter.split_text(text)
                for i, piece in enumerate(pieces):
                    batch.append(Document(
                        id=f"{doc_id}#{i}",
                        page_content=piece,
                        metadata={"doc_id": doc_id, "title": title, "chunk": i, "chunks": len(pieces)},
                    ))
                documents += 1
                if len(batch) >= self.settings.embed_batch_size:
                    flush()
            flush()
            if documents == 0:
                raise DatasetError(f"{dataset_path.name} contains no usable records (need an 'abstract' field)")
        except BaseException:
            keywords.close()
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass
            (self.root / keyword_db).unlink(missing_ok=True)
            raise

        sample = self.client.get_collection(collection_name).get(limit=1, include=["embeddings"])
        dimensions = len(sample["embeddings"][0]) if len(sample["embeddings"]) else None
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "collection": collection_name,
            "keyword_db": keyword_db,
            "dataset": {
                "path": str(dataset_path),
                "name": dataset_path.name,
                "sha256": dataset_sha,
                "size_bytes": dataset_path.stat().st_size,
                "records": loader.stats.lines,
                "documents": documents,
                "duplicates": duplicates,
                "malformed": loader.stats.malformed,
                "missing_text": loader.stats.missing_text,
            },
            "chunks": chunks,
            "embedding": {
                "model": self.settings.embed_model,
                "dimensions": dimensions,
                "doc_prefix": self.embeddings.doc_prefix,
                "query_prefix": self.embeddings.query_prefix,
            },
            "chunking": {"chunk_size": self.settings.chunk_size, "chunk_overlap": self.settings.chunk_overlap},
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "build_seconds": round(time.monotonic() - started, 1),
        }
        self._write_manifest(manifest)  # commit point
        log.info(
            "Indexed %d documents (%d chunks, %d duplicates, %d malformed) in %.1fs",
            documents, chunks, duplicates, loader.stats.malformed, manifest["build_seconds"],
        )
        return ActiveIndex(manifest=manifest, vectorstore=vectorstore, keywords=keywords, embeddings=self.embeddings)

    def _cleanup(self) -> None:
        """Delete every collection / keyword file that does not belong to the active index."""
        active = self._active
        if active is None:
            return
        keep_collection = active.manifest["collection"]
        keep_db = active.manifest["keyword_db"]
        try:
            for collection in self.client.list_collections():
                name = collection if isinstance(collection, str) else collection.name
                if name != keep_collection:
                    self.client.delete_collection(name)
                    log.info("Discarded old vector collection %s", name)
        except Exception as exc:
            log.warning("Could not clean old collections: %s", exc)
        for path in self.root.glob("keywords-*.sqlite*"):
            if not path.name.startswith(keep_db):
                try:
                    path.unlink()
                except OSError as exc:
                    log.debug("Could not delete %s: %s", path, exc)
