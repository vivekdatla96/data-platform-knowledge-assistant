"""Local vector store: SQLite for documents, chunks and provenance; NumPy for similarity.

Embeddings are stored as float32 BLOBs next to the chunk text and provenance, so the whole index
is a single file that can be copied, inspected with any SQLite client and rebuilt from source.
Search is exact (brute-force cosine), which is fast at this scale (hundreds to low tens of
thousands of chunks). See docs/ARCHITECTURE.md for when to move to an ANN index.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dpka.models import Chunk

SCHEMA_VERSION = "1"

DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    doc_type      TEXT NOT NULL,
    format_label  TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    chunk_count   INTEGER NOT NULL,
    ingested_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal       INTEGER NOT NULL,
    locator       TEXT NOT NULL,
    title         TEXT NOT NULL,
    text          TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    doc_type      TEXT NOT NULL,
    start_line    INTEGER,
    end_line      INTEGER,
    suspicious    INTEGER NOT NULL DEFAULT 0,
    embedding     BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);
"""


class EmbeddingModelMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    doc_type: str
    format_label: str
    content_hash: str
    chunk_count: int
    ingested_at: str


class VectorStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(DDL)
        self._set_meta_default("schema_version", SCHEMA_VERSION)

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ meta
    def _set_meta_default(self, key: str, value: str) -> None:
        self._conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)", (key, value))
        self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def bind_embedding_model(self, name: str, dim: int) -> None:
        """Record which model built the index; refuse to mix vectors from different models."""
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError("embedding dimension must be a positive integer")
        existing = self.get_meta("embedding_model")
        if existing is None:
            with self.transaction():
                self._conn.execute("INSERT INTO meta VALUES ('embedding_model', ?)", (name,))
                self._conn.execute("INSERT INTO meta VALUES ('embedding_dim', ?)", (str(dim),))
        elif existing != name or self.get_meta("embedding_dim") != str(dim):
            raise EmbeddingModelMismatch(
                f"index was built with {existing!r} ({self.get_meta('embedding_dim')} dimensions) "
                f"but current embedder is {name!r} ({dim} dimensions); "
                "re-run ingestion with --rebuild"
            )

    def reset(self) -> None:
        with self.transaction():
            self._conn.execute("DELETE FROM chunks")
            self._conn.execute("DELETE FROM documents")
            self._conn.execute("DELETE FROM meta WHERE key IN ('embedding_model', 'embedding_dim')")

    # ------------------------------------------------------------ documents
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise

    def documents(self) -> dict[str, DocumentRecord]:
        rows = self._conn.execute(
            "SELECT doc_id, doc_type, format_label, content_hash, chunk_count, ingested_at "
            "FROM documents"
        ).fetchall()
        return {r[0]: DocumentRecord(*r) for r in rows}

    def chunk_embeddings_for(self, doc_id: str) -> dict[str, tuple[str, np.ndarray]]:
        """chunk_id -> (content_hash, embedding) for reuse when a document is re-ingested."""
        rows = self._conn.execute(
            "SELECT chunk_id, content_hash, embedding FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        return {r[0]: (r[1], np.frombuffer(r[2], dtype=np.float32)) for r in rows}

    def replace_document(
        self,
        record: DocumentRecord,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        """Atomically replace a document and all of its chunks."""
        dimension = self.get_meta("embedding_dim")
        if dimension is None:
            raise ValueError("bind an embedding model before storing documents")
        with np.errstate(over="ignore", invalid="ignore"):
            embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.shape != (len(chunks), int(dimension)):
            raise ValueError("embeddings must have shape (chunk count, bound embedding dimension)")
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must contain only finite float32 values")
        if record.chunk_count != len(chunks):
            raise ValueError("document chunk count must match supplied chunks")
        if any(chunk.doc_id != record.doc_id for chunk in chunks):
            raise ValueError("all chunks must belong to the supplied document")
        with self.transaction() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (record.doc_id,))
            conn.execute(
                "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.doc_id,
                    record.doc_type,
                    record.format_label,
                    record.content_hash,
                    record.chunk_count,
                    record.ingested_at,
                ),
            )
            conn.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        c.chunk_id,
                        c.doc_id,
                        c.ordinal,
                        c.locator,
                        c.title,
                        c.text,
                        c.content_hash,
                        c.doc_type,
                        c.start_line,
                        c.end_line,
                        int(c.suspicious),
                        np.asarray(e, dtype=np.float32).tobytes(),
                    )
                    for c, e in zip(chunks, embeddings, strict=True)
                ],
            )

    def delete_document(self, doc_id: str) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    # ----------------------------------------------------------------- read
    def load_all(self) -> tuple[list[Chunk], np.ndarray]:
        rows = self._conn.execute(
            "SELECT chunk_id, doc_id, ordinal, locator, title, text, content_hash, doc_type, "
            "start_line, end_line, suspicious, embedding FROM chunks ORDER BY doc_id, ordinal"
        ).fetchall()
        chunks = [
            Chunk(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], bool(r[10]))
            for r in rows
        ]
        if not rows:
            return chunks, np.zeros((0, 0), dtype=np.float32)
        matrix = np.stack([np.frombuffer(r[11], dtype=np.float32) for r in rows])
        return chunks, matrix

    def stats(self) -> dict[str, object]:
        n_docs = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        n_chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        by_type = dict(
            self._conn.execute("SELECT doc_type, COUNT(*) FROM chunks GROUP BY doc_type").fetchall()
        )
        return {
            "documents": n_docs,
            "chunks": n_chunks,
            "chunks_by_type": by_type,
            "embedding_model": self.get_meta("embedding_model"),
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
