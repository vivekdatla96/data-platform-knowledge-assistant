"""Core data structures shared by ingestion, retrieval and generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def make_chunk_id(doc_id: str, locator: str) -> str:
    """Deterministic chunk id.

    Derived from *where* the chunk lives (document + structural locator), not from its text, so
    the id is stable across edits. Content changes are tracked separately via ``content_hash``.
    """
    return sha256_hex(f"{doc_id}::{locator}")[:20]


@dataclass(frozen=True)
class ChunkDraft:
    """A chunk as produced by a loader, before ids and hashes are assigned."""

    locator: str
    title: str
    text: str
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    ordinal: int
    locator: str
    title: str
    text: str
    content_hash: str
    doc_type: str
    start_line: int | None = None
    end_line: int | None = None
    suspicious: bool = False

    @property
    def citation_label(self) -> str:
        lines = ""
        if self.start_line is not None:
            lines = f" (lines {self.start_line}-{self.end_line})"
        return f"{self.doc_id} :: {self.locator}{lines}"


@dataclass(frozen=True)
class ParsedDocument:
    doc_id: str  # path relative to the corpus root, POSIX style
    doc_type: str  # snowflake_metadata | snowflake_lineage | informatica_mapping | runbook
    format_label: str  # human-readable label shown in the UI, always says "synthetic/simplified"
    content_hash: str
    chunks: list[ChunkDraft]


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    chunks_embedded: int = 0
    chunks_reused: int = 0

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        return (
            f"added={len(self.added)} updated={len(self.updated)} "
            f"unchanged={len(self.unchanged)} deleted={len(self.deleted)} "
            f"failed={len(self.failed)} chunks_embedded={self.chunks_embedded} "
            f"chunks_reused={self.chunks_reused}"
        )
