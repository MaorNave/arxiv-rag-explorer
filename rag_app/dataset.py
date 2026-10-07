"""Dataset handling: locating DATA_PATH, streaming the .jsonl file, fingerprinting it.

The file is never loaded into memory as a whole: records are parsed line by line
and handed to the indexer in small batches.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from .config import PROJECT_ROOT

log = logging.getLogger(__name__)

# The assignment's schema is {id, title, abstract, ...}; a few common aliases are
# accepted so that "any .jsonl dataset" with a similar shape still works.
TEXT_FIELDS = ("abstract", "text", "content", "summary", "body")
ID_FIELDS = ("id", "doc_id", "paper_id", "arxiv_id", "_id", "uuid")
TITLE_FIELDS = ("title", "name", "headline")

_ARXIV_NEW = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_ARXIV_OLD = re.compile(r"^[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$")


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be located or read."""


def is_url(value: str) -> bool:
    return value.lower().startswith(("http://", "https://"))


def arxiv_url(doc_id: str) -> str | None:
    """Link to the paper page when the id looks like an arXiv identifier."""
    if _ARXIV_NEW.match(doc_id) or _ARXIV_OLD.match(doc_id):
        return f"https://arxiv.org/abs/{doc_id}"
    return None


def _newest_jsonl(directory: Path) -> Path | None:
    files = [p for p in directory.glob("*.jsonl") if p.is_file()]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def resolve_dataset_path(data_path: str, downloads_dir: Path | None = None) -> Path:
    """Turn DATA_PATH into an existing .jsonl file.

    * an existing file is used as is;
    * a directory resolves to the most recently modified ``*.jsonl`` inside it;
    * a missing file falls back to a ``*.jsonl`` in the same directory (handy when
      the mounted file name differs from DATA_PATH, e.g. arxiv_5k vs arxiv_2.9k);
    * an http(s) URL is downloaded (streamed) into ``downloads_dir``.
    """
    if is_url(data_path):
        if downloads_dir is None:
            raise DatasetError("A downloads directory is required for URL datasets")
        return download_dataset(data_path, downloads_dir)

    requested = Path(data_path).expanduser()
    candidates = [requested] if requested.is_absolute() else [Path.cwd() / requested, PROJECT_ROOT / requested]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
        if candidate.is_dir():
            found = _newest_jsonl(candidate)
            if found:
                log.info("DATA_PATH %s is a directory; using %s", candidate, found.name)
                return found.resolve()

    for candidate in candidates:
        if candidate.parent.is_dir():
            found = _newest_jsonl(candidate.parent)
            if found:
                log.warning("DATA_PATH %s does not exist; falling back to %s", data_path, found)
                return found.resolve()

    raise DatasetError(f"Dataset not found: {data_path!r} (no .jsonl file at that path or next to it)")


def download_dataset(url: str, downloads_dir: Path) -> Path:
    """Stream a remote .jsonl file to disk; reuse the cached copy if the download fails."""
    downloads_dir.mkdir(parents=True, exist_ok=True)
    name = Path(urlparse(url).path).name or "dataset.jsonl"
    if not name.endswith(".jsonl"):
        name += ".jsonl"
    url_hash = hashlib.sha1(url.encode()).hexdigest()[:10]
    target = downloads_dir / f"{url_hash}-{name}"
    partial = target.with_suffix(".part")
    try:
        log.info("Downloading dataset from %s", url)
        with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
            response.raise_for_status()
            with partial.open("wb") as fh:
                for block in response.iter_bytes(1 << 20):
                    fh.write(block)
        partial.replace(target)
    except (httpx.HTTPError, OSError) as exc:
        partial.unlink(missing_ok=True)
        if target.exists():
            log.warning("Download failed (%s); using cached copy %s", exc, target)
        else:
            raise DatasetError(f"Could not download dataset from {url}: {exc}") from exc
    return target


def file_sha256(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def count_records(path: Path) -> int:
    """Number of non-blank lines (streamed; used for progress reporting)."""
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def _first_text(record: dict, fields: tuple[str, ...]) -> str:
    for name in fields:
        value = record.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


@dataclass
class LoadStats:
    lines: int = 0
    loaded: int = 0
    malformed: int = 0
    missing_text: int = 0


class JsonlAbstractLoader(BaseLoader):
    """LangChain document loader that streams abstracts from a .jsonl file.

    Each record becomes one ``Document`` whose ``page_content`` is the abstract
    (the only field that is embedded) and whose metadata keeps ``doc_id`` and
    ``title`` for citations. All other fields are dropped.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.stats = LoadStats()

    def lazy_load(self) -> Iterator[Document]:
        self.stats = LoadStats()
        with self.path.open("r", encoding="utf-8-sig", errors="replace") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                self.stats.lines += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    self.stats.malformed += 1
                    continue
                if not isinstance(record, dict):
                    self.stats.malformed += 1
                    continue
                text = _first_text(record, TEXT_FIELDS)
                if not text:
                    self.stats.missing_text += 1
                    continue
                doc_id = normalize_ws(_first_text(record, ID_FIELDS)) or f"line-{line_no}"
                title = normalize_ws(_first_text(record, TITLE_FIELDS)) or doc_id
                self.stats.loaded += 1
                yield Document(
                    page_content=normalize_ws(text),
                    metadata={"doc_id": doc_id, "title": title},
                )


def validate_jsonl_sample(path: Path, max_lines: int = 50) -> None:
    """Cheap sanity check for uploaded files: the first records must be usable."""
    checked = usable = 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            checked += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = None
            if isinstance(record, dict) and _first_text(record, TEXT_FIELDS):
                usable += 1
            if checked >= max_lines:
                break
    if checked == 0:
        raise DatasetError("The file is empty")
    if usable == 0:
        raise DatasetError(
            "No usable records found: every line must be a JSON object with an "
            f"'abstract' field (or one of {', '.join(TEXT_FIELDS[1:])})"
        )
