"""Sparse keyword index (BM25) on SQLite FTS5.

It complements dense vector search with exact-term matching, which matters for
acronyms, model names and rare technical terms ("RLBFF", "SD3.5-Flash", ...).
SQLite ships with Python, works on disk and needs no extra service.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are aren as at be because been before being
    below between both but by can cannot could did do does doing don down during each either else even ever
    every few for from further get gets got had has have having he her here hers herself him himself his how
    however i if in into is isn it its itself just let me might more most must my myself no nor not now of
    off on once only or other our ours ourselves out over own per rather same shall she should since so some
    such than that the their theirs them themselves then there these they this those through thus to too
    under until up upon us very via was we were what when where whether which while who whom whose why will
    with within without would yet you your yours yourself yourselves
    describe explain tell give show list compare compared versus vs please paper papers abstract abstracts
    research study studies work works recent recently new novel approach approaches method methods propose
    proposed use used using based kind kinds way ways help helps make makes
    """.split()
)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]


def build_match_query(texts: Iterable[str], phrases: Iterable[str] = (), max_terms: int = 40) -> str | None:
    """Build an FTS5 MATCH expression: quoted terms/phrases joined with OR.

    Quoting makes user input inert (no FTS operators can be injected) while still
    letting the tokenizer stem each term. Multi-word keywords are added as phrases
    so documents containing the exact phrase get an extra BM25 boost.
    """
    parts: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        words = _TOKEN_RE.findall(phrase.lower())
        if 1 < len(words) <= 4:
            quoted = '"' + " ".join(words) + '"'
            if quoted not in seen:
                seen.add(quoted)
                parts.append(quoted)
    for text in texts:
        for token in _tokens(text):
            quoted = f'"{token}"'
            if quoted not in seen:
                seen.add(quoted)
                parts.append(quoted)
    return " OR ".join(parts[:max_terms]) or None


@dataclass
class KeywordHit:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float  # BM25 (SQLite convention: lower is better)


class KeywordIndex:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._con: sqlite3.Connection | None = None

    def _connection(self) -> sqlite3.Connection:
        if self._con is None:
            self._con = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        return self._con

    def create(self) -> None:
        with self._lock:
            con = self._connection()
            con.executescript(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                    chunk_id UNINDEXED, doc_id UNINDEXED, title, body,
                    tokenize = 'porter unicode61 remove_diacritics 2'
                );
                CREATE TABLE IF NOT EXISTS docs (doc_id TEXT PRIMARY KEY, title TEXT NOT NULL);
                """
            )
            con.commit()

    def register_doc(self, doc_id: str, title: str) -> bool:
        """Record a document id; returns False if it was already indexed (duplicate)."""
        with self._lock:
            cur = self._connection().execute(
                "INSERT OR IGNORE INTO docs(doc_id, title) VALUES (?, ?)", (doc_id, title)
            )
            return cur.rowcount == 1

    def add_chunks(self, rows: Iterable[tuple[str, str, str, str]]) -> None:
        """rows: (chunk_id, doc_id, title, text)."""
        with self._lock:
            con = self._connection()
            con.executemany("INSERT INTO chunks(chunk_id, doc_id, title, body) VALUES (?, ?, ?, ?)", rows)
            con.commit()

    def count_chunks(self) -> int:
        with self._lock:
            return self._connection().execute("SELECT count(*) FROM chunks").fetchone()[0]

    def search(self, match_query: str, k: int) -> list[KeywordHit]:
        with self._lock:
            try:
                rows = self._connection().execute(
                    """
                    SELECT chunk_id, doc_id, title, body, bm25(chunks, 0.0, 0.0, 2.0, 1.0) AS score
                    FROM chunks WHERE chunks MATCH ? ORDER BY score LIMIT ?
                    """,
                    (match_query, k),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [KeywordHit(*row) for row in rows]

    def random_titles(self, n: int) -> list[str]:
        with self._lock:
            rows = self._connection().execute(
                "SELECT title FROM docs ORDER BY random() LIMIT ?", (n,)
            ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        with self._lock:
            if self._con is not None:
                self._con.commit()
                self._con.close()
                self._con = None
