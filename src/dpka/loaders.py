"""Parsers and validators for the three supported source formats.

All three formats are simplified, synthetic stand-ins defined by this project:

* ``dpka.snowflake_metadata.v1`` / ``dpka.snowflake_lineage.v1`` (JSON): loosely modeled on
  Snowflake INFORMATION_SCHEMA and lineage views. Not a real Snowflake export.
* ``dpka.informatica_mapping.v1`` (XML): loosely modeled on a PowerCenter mapping export. Not
  importable into Informatica.
* Markdown runbooks.

Each loader validates its input and returns structure-aware chunks with stable locators.
"""

from __future__ import annotations

import json
import re
from typing import Literal
from xml.etree.ElementTree import Element

from defusedxml import ElementTree as SafeET
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from dpka.models import ChunkDraft, ParsedDocument, sha256_hex

MAX_FILE_BYTES = 2_000_000
MAX_CHUNK_CHARS = 1800
IDENT = re.compile(r"^[A-Z_][A-Z0-9_$]*$")  # unquoted Snowflake identifier, upper case

FORMAT_LABELS = {
    "snowflake_metadata": "Synthetic Snowflake metadata (simplified JSON, not a real export)",
    "snowflake_lineage": "Synthetic lineage (simplified JSON, hand-written)",
    "informatica_mapping": "Synthetic Informatica-style mapping (simplified XML, not importable)",
    "runbook": "Synthetic runbook (Markdown)",
}


class DocumentValidationError(ValueError):
    """Raised when a source file fails validation. The message is shown in ingest reports."""


# --------------------------------------------------------------------------- Snowflake (JSON)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ColumnMeta(_Strict):
    name: str
    data_type: str
    nullable: bool
    comment: str = ""

    @field_validator("name")
    @classmethod
    def _ident(cls, v: str) -> str:
        if not IDENT.match(v):
            raise ValueError(f"invalid column identifier {v!r}")
        return v


class TableMeta(_Strict):
    schema_: str = Field(alias="schema")
    name: str
    table_type: Literal["BASE TABLE", "VIEW"]
    comment: str = ""
    load_pattern: str = ""
    owner_role: str = ""
    clustering_key: str | None = None
    columns: list[ColumnMeta] = Field(min_length=1)

    @field_validator("schema_", "name")
    @classmethod
    def _ident(cls, v: str) -> str:
        if not IDENT.match(v):
            raise ValueError(f"invalid identifier {v!r}")
        return v

    @property
    def fqn(self) -> str:
        return f"{self.schema_}.{self.name}"


class SnowflakeMetadataFile(_Strict):
    format: Literal["dpka.snowflake_metadata.v1"]
    note: str
    database: str
    tables: list[TableMeta] = Field(min_length=1)


class ColumnLineage(_Strict):
    source: str
    target: str


class LineageEdge(_Strict):
    source: str
    target: str
    process: str
    columns: list[ColumnLineage] = Field(default_factory=list)


class SnowflakeLineageFile(_Strict):
    format: Literal["dpka.snowflake_lineage.v1"]
    note: str
    database: str
    edges: list[LineageEdge] = Field(min_length=1)


def _format_validation_error(err: ValidationError) -> str:
    first = err.errors()[0]
    loc = ".".join(str(p) for p in first["loc"])
    return f"{loc}: {first['msg']} ({err.error_count()} error(s))"


def parse_snowflake_json(doc_id: str, raw: bytes) -> ParsedDocument:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DocumentValidationError(f"not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise DocumentValidationError("top-level JSON value must be an object")

    fmt = data.get("format")
    try:
        if fmt == "dpka.snowflake_metadata.v1":
            return _metadata_doc(doc_id, raw, SnowflakeMetadataFile.model_validate(data))
        if fmt == "dpka.snowflake_lineage.v1":
            return _lineage_doc(doc_id, raw, SnowflakeLineageFile.model_validate(data))
    except ValidationError as exc:
        raise DocumentValidationError(_format_validation_error(exc)) from exc
    raise DocumentValidationError(f"unknown or missing 'format': {fmt!r}")


def _metadata_doc(doc_id: str, raw: bytes, meta: SnowflakeMetadataFile) -> ParsedDocument:
    seen: set[str] = set()
    chunks = []
    for t in meta.tables:
        if t.fqn in seen:
            raise DocumentValidationError(f"duplicate table {t.fqn}")
        seen.add(t.fqn)
        lines = [
            f"Snowflake {t.table_type.lower()} {meta.database}.{t.fqn}",
            f"Description: {t.comment}",
        ]
        if t.load_pattern:
            lines.append(f"Load pattern: {t.load_pattern}")
        if t.owner_role:
            lines.append(f"Owner role: {t.owner_role}")
        if t.clustering_key:
            lines.append(f"Clustering key: {t.clustering_key}")
        lines.append("Columns:")
        for c in t.columns:
            null = "NULL" if c.nullable else "NOT NULL"
            lines.append(f"- {c.name} {c.data_type} {null}: {c.comment}".rstrip(": "))
        chunks.append(ChunkDraft(locator=f"table:{t.fqn}", title=t.fqn, text="\n".join(lines)))
    return ParsedDocument(
        doc_id, "snowflake_metadata", FORMAT_LABELS["snowflake_metadata"], sha256_hex(raw), chunks
    )


def _lineage_doc(doc_id: str, raw: bytes, lin: SnowflakeLineageFile) -> ParsedDocument:
    seen: set[str] = set()
    chunks = []
    for e in lin.edges:
        locator = f"lineage:{e.source}->{e.target}"
        if locator in seen:
            raise DocumentValidationError(f"duplicate lineage edge {e.source} -> {e.target}")
        seen.add(locator)
        lines = [
            f"Lineage: {e.source} feeds {e.target}",
            f"Upstream {e.source} feeds downstream {e.target} through {e.process}.",
        ]
        if e.columns:
            lines.append("Column lineage:")
            lines += [f"- {e.source}.{c.source} -> {e.target}.{c.target}" for c in e.columns]
        chunks.append(
            ChunkDraft(locator=locator, title=f"{e.source} -> {e.target}", text="\n".join(lines))
        )
    return ParsedDocument(
        doc_id, "snowflake_lineage", FORMAT_LABELS["snowflake_lineage"], sha256_hex(raw), chunks
    )


# ------------------------------------------------------------------- Informatica-style (XML)


def _req(el: Element, attr: str, where: str) -> str:
    val = (el.get(attr) or "").strip()
    if not val:
        raise DocumentValidationError(f"{where}: missing required attribute {attr}")
    return val


def parse_informatica_xml(doc_id: str, raw: bytes) -> ParsedDocument:
    try:
        # defusedxml rejects entity expansion and external entities (untrusted input).
        root = SafeET.fromstring(raw)
    except SafeET.ParseError as exc:
        raise DocumentValidationError(f"malformed XML: {exc}") from exc
    except Exception as exc:  # defusedxml raises its own DTD/entity errors
        raise DocumentValidationError(f"rejected XML: {exc.__class__.__name__}") from exc

    if root.tag != "MAPPING_EXPORT" or root.get("FORMAT") != "dpka.informatica_mapping.v1":
        raise DocumentValidationError(
            "root must be <MAPPING_EXPORT FORMAT='dpka.informatica_mapping.v1'>"
        )
    mappings = root.findall("MAPPING")
    if not mappings:
        raise DocumentValidationError("no <MAPPING> elements")

    chunks: list[ChunkDraft] = []
    for m in mappings:
        name = _req(m, "NAME", "MAPPING")
        where = f"MAPPING {name}"
        sources = [_req(s, "NAME", where) for s in m.findall("SOURCE")]
        targets = m.findall("TARGET")
        if not sources or not targets:
            raise DocumentValidationError(f"{where}: needs at least one SOURCE and one TARGET")
        fields = m.findall("FIELD_MAPPING")
        if not fields:
            raise DocumentValidationError(f"{where}: no FIELD_MAPPING elements")

        # Overview: what, where, how incremental, when it runs, what happens on failure.
        lines = [
            f"Informatica-style mapping {name} (folder {root.get('FOLDER', '?')})",
            f"Description: {m.get('DESCRIPTION', '')}",
            f"Sources: {', '.join(sources)}",
        ]
        for t in targets:
            lines.append(
                f"Target: {_req(t, 'NAME', where)} load type {t.get('LOAD_TYPE', '?')}"
                f" keyed on {t.get('KEY', '?')}"
            )
        for v in m.findall("MAPPING_VARIABLE"):
            lines.append(f"Mapping variable {v.get('NAME')}: {v.get('DESCRIPTION', '')}")
        sf = m.find("SOURCE_FILTER")
        if sf is not None and (sf.text or "").strip():
            lines.append(f"Source filter (incremental extract): {sf.text.strip()}")
        for s in m.findall("SESSION"):
            lines.append(
                f"Session {s.get('NAME')} in workflow {s.get('WORKFLOW')}, schedule "
                f"{s.get('SCHEDULE')}."
            )
            for attr, label in (
                ("PRE_SQL", "Pre-session"),
                ("POST_SQL", "Post-session"),
                ("ON_FAILURE", "On failure"),
            ):
                if s.get(attr):
                    lines.append(f"{label}: {s.get(attr)}")
        chunks.append(ChunkDraft(f"mapping:{name}#overview", f"{name} overview", "\n".join(lines)))

        trans = m.findall("TRANSFORMATION")
        if trans:
            tl = [f"Transformations in mapping {name}:"]
            for t in trans:
                tl.append(
                    f"- {_req(t, 'NAME', where)} ({t.get('TYPE', '?')}): "
                    f"{t.get('DESCRIPTION', '')}".rstrip(": ")
                )
                for p in t.findall("PORT"):
                    tl.append(f"  - port {p.get('NAME')} = {p.get('EXPRESSION')}")
            chunks.append(
                ChunkDraft(
                    f"mapping:{name}#transformations", f"{name} transformations", "\n".join(tl)
                )
            )

        target_names = ", ".join(t.get("NAME", "") for t in targets)
        fl = [
            f"Source-to-target field mapping for {name} ({', '.join(sources)} -> {target_names}):"
        ]
        for f in fields:
            src = _req(f, "SOURCE_FIELD", where)
            tgt = _req(f, "TARGET_FIELD", where)
            fl.append(f"- {src} -> {tgt}: {f.get('LOGIC', '')}".rstrip(": "))
        chunks.append(
            ChunkDraft(f"mapping:{name}#field_mappings", f"{name} field mappings", "\n".join(fl))
        )

    _check_unique(chunks)
    return ParsedDocument(
        doc_id, "informatica_mapping", FORMAT_LABELS["informatica_mapping"], sha256_hex(raw), chunks
    )


# ------------------------------------------------------------------------------ Markdown

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


def parse_markdown(doc_id: str, raw: bytes) -> ParsedDocument:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentValidationError("runbook is not valid UTF-8") from exc
    lines = text.splitlines()
    if not text.strip():
        raise DocumentValidationError("empty document")

    title = None
    sections: list[tuple[list[str], int, list[str]]] = []  # (heading path, start line, body)
    path: list[str] = []
    current: tuple[list[str], int, list[str]] = ([], 1, [])
    in_fence = False
    for i, line in enumerate(lines, start=1):
        if FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else HEADING.match(line)
        if m:
            level, heading = len(m.group(1)), m.group(2).strip()
            if level == 1 and title is None:
                title = heading
            sections.append(current)
            path = path[: max(level - 2, 0)] + [heading] if level > 1 else []
            current = (list(path), i, [])
        else:
            current[2].append(line)
    sections.append(current)

    if title is None:
        raise DocumentValidationError("runbook must start with a '# Title' heading")

    chunks: list[ChunkDraft] = []
    used: dict[str, int] = {}
    for heading_path, start, body in sections:
        content = [b for b in body if b.strip()]
        # skip empty sections and callout-only notes (e.g. the "> Synthetic runbook" banner)
        if not content or all(b.lstrip().startswith(">") for b in content):
            continue
        base = "section:" + (" > ".join(heading_path) if heading_path else "(intro)")
        n = used.get(base, 0)
        used[base] = n + 1
        locator = base if n == 0 else f"{base} [{n + 1}]"
        crumb = " > ".join([title, *heading_path])
        for part, (p_start, p_body) in enumerate(_split_long(start, body)):
            loc = locator if part == 0 else f"{locator} #part{part + 1}"
            chunks.append(
                ChunkDraft(
                    locator=loc,
                    title=crumb,
                    text=f"{crumb}\n\n" + "\n".join(p_body).strip(),
                    start_line=p_start,
                    end_line=p_start + len(p_body),
                )
            )
    _check_unique(chunks)
    return ParsedDocument(doc_id, "runbook", FORMAT_LABELS["runbook"], sha256_hex(raw), chunks)


def _split_long(heading_line: int, body: list[str]) -> list[tuple[int, list[str]]]:
    """Split a section body on blank lines when it exceeds MAX_CHUNK_CHARS."""
    if sum(len(b) + 1 for b in body) <= MAX_CHUNK_CHARS:
        return [(heading_line, body)]
    parts: list[tuple[int, list[str]]] = []
    cur: list[str] = []
    cur_start = heading_line
    size = 0
    for offset, line in enumerate(body, start=1):
        if size + len(line) > MAX_CHUNK_CHARS and cur and not line.strip():
            parts.append((cur_start, cur))
            cur, size, cur_start = [], 0, heading_line + offset
            continue
        cur.append(line)
        size += len(line) + 1
    if cur:
        parts.append((cur_start, cur))
    return parts


def _check_unique(chunks: list[ChunkDraft]) -> None:
    seen: set[str] = set()
    for c in chunks:
        if c.locator in seen:
            raise DocumentValidationError(f"duplicate locator {c.locator}")
        seen.add(c.locator)


# ------------------------------------------------------------------------------- dispatch

PARSERS = {
    ".json": parse_snowflake_json,
    ".xml": parse_informatica_xml,
    ".md": parse_markdown,
}


def parse_document(doc_id: str, raw: bytes) -> ParsedDocument:
    if len(raw) > MAX_FILE_BYTES:
        raise DocumentValidationError(f"file larger than {MAX_FILE_BYTES} bytes")
    suffix = "." + doc_id.rsplit(".", 1)[-1].lower() if "." in doc_id else ""
    parser = PARSERS.get(suffix)
    if parser is None:
        raise DocumentValidationError(f"unsupported file type {suffix!r}")
    doc = parser(doc_id, raw)
    if not doc.chunks:
        raise DocumentValidationError("document produced no chunks")
    return doc
