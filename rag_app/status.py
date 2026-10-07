"""Thread-safe application status shared by the bootstrapper, the indexer and the API."""

from __future__ import annotations

import threading
import time
from typing import Any

# Lifecycle: starting -> waiting_for_ollama -> pulling_models -> indexing -> ready
#            (any state may end in "error"; a dataset change moves ready -> indexing)
STATES = ("starting", "waiting_for_ollama", "pulling_models", "indexing", "ready", "error")


class StatusTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {
            "state": "starting",
            "message": "Starting up…",
            "progress": None,
            "error": None,
            "updated_at": time.time(),
        }

    def update(self, **fields: Any) -> None:
        with self._lock:
            self._data.update(fields)
            self._data["updated_at"] = time.time()

    def set_state(self, state: str, message: str, *, progress: dict | None = None, error: str | None = None) -> None:
        self.update(state=state, message=message, progress=progress, error=error)

    @property
    def state(self) -> str:
        with self._lock:
            return self._data["state"]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)
