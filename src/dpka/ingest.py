"""Repeatable, incremental ingestion of a corpus directory into the vector store.

The corpus directory is only ever read. Each run compares files on disk with the documents
already in the index:

* new file                      -> parsed, validated, embedded, inserted
* same content hash             -> skipped (no parsing, no embedding)
* changed content hash          -> re-parsed; chunks whose id and text are unchanged reuse their
                                   stored embedding; the document is replaced atomically
* file removed from disk        -> document and all of its chunks deleted
* file fails validation         -> reported; a previously indexed version is kept (last known
                                   good) so a bad edit cannot silently empty the index
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from dpka.embeddings import Embedder
from dpka.loaders import PARSERS, DocumentValidationError, parse_document
from dpka.models import Chunk, IngestReport, ParsedDocument, make_chunk_id, sha256_hex
from dpka.safety import is_suspicious
from dpka.store import DocumentRecord, VectorStore, utc_now

log = logging.getLogger(__name__)


def discover(corpus_dir: Path) -> dict[str, Path]:
    """doc_id (POSIX path relative to corpus root) -> file path. Ignores hidden files/symlinks."""
    corpus_dir = corpus_dir.absolute()
    found: dict[str, Path] = {}
    for path in sorted(corpus_dir.rglob("*")):
        rel = path.relative_to(corpus_dir)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix.lower() in PARSERS:
            found[rel.as_posix()] = path
    return found


def build_chunks(doc: ParsedDocument) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=make_chunk_id(doc.doc_id, d.locator),
            doc_id=doc.doc_id,
            ordinal=i,
            locator=d.locator,
            title=d.title,
            text=d.text,
            content_hash=sha256_hex(d.text),
            doc_type=doc.doc_type,
            start_line=d.start_line,
            end_line=d.end_line,
            suspicious=is_suspicious(d.text),
        )
        for i, d in enumerate(doc.chunks)
    ]


def ingest(
    corpus_dir: Path,
    store: VectorStore,
    embedder: Embedder,
    *,
    rebuild: bool = False,
) -> IngestReport:
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")
    if rebuild:
        store.reset()
    store.bind_embedding_model(embedder.name, embedder.dim)

    report = IngestReport()
    on_disk = discover(corpus_dir)
    indexed = store.documents()

    for doc_id, path in on_disk.items():
        raw = path.read_bytes()
        file_hash = sha256_hex(raw)
        previous = indexed.get(doc_id)
        if previous is not None and previous.content_hash == file_hash:
            report.unchanged.append(doc_id)
            continue
        try:
            doc = parse_document(doc_id, raw)
        except DocumentValidationError as exc:
            report.failed[doc_id] = str(exc)
            log.warning("validation failed for %s: %s", doc_id, exc)
            continue

        chunks = build_chunks(doc)
        reusable = store.chunk_embeddings_for(doc_id) if previous else {}
        vectors: list[np.ndarray | None] = []
        to_embed: list[int] = []
        for i, c in enumerate(chunks):
            hit = reusable.get(c.chunk_id)
            if hit is not None and hit[0] == c.content_hash:
                vectors.append(hit[1])
                report.chunks_reused += 1
            else:
                vectors.append(None)
                to_embed.append(i)
        if to_embed:
            new = embedder.embed_documents([chunks[i].text for i in to_embed])
            for i, v in zip(to_embed, new, strict=True):
                vectors[i] = v
            report.chunks_embedded += len(to_embed)

        store.replace_document(
            DocumentRecord(
                doc_id, doc.doc_type, doc.format_label, file_hash, len(chunks), utc_now()
            ),
            chunks,
            np.stack(vectors),  # type: ignore[arg-type]
        )
        (report.updated if previous else report.added).append(doc_id)

    for doc_id in sorted(set(indexed) - set(on_disk)):
        store.delete_document(doc_id)
        report.deleted.append(doc_id)

    return report
