from __future__ import annotations

import json

import pytest
from conftest import CORPUS

from dpka.ingest import build_chunks
from dpka.loaders import DocumentValidationError, parse_document
from dpka.models import make_chunk_id


def _parse(rel: str):
    return parse_document(rel, (CORPUS / rel).read_bytes())


def test_snowflake_metadata_one_chunk_per_table():
    doc = _parse("snowflake/dw_tables.json")
    assert doc.doc_type == "snowflake_metadata"
    assert "Synthetic" in doc.format_label
    locators = [c.locator for c in doc.chunks]
    assert "table:DW.FACT_ORDERS" in locators
    assert "table:AUDIT.ETL_CONTROL" in locators
    fact = next(c for c in doc.chunks if c.locator == "table:DW.FACT_ORDERS")
    assert "NET_AMOUNT NUMBER(12,2) NOT NULL" in fact.text


def test_lineage_edges_become_chunks():
    doc = _parse("snowflake/lineage.json")
    assert doc.doc_type == "snowflake_lineage"
    assert "lineage:DW.FACT_ORDERS->MART.DAILY_SALES" in [c.locator for c in doc.chunks]


def test_informatica_mapping_has_overview_transformations_and_field_mappings():
    doc = _parse("informatica/m_ORDERS_STG_TO_FACT_ORDERS.xml")
    by_loc = {c.locator: c for c in doc.chunks}
    assert set(by_loc) == {
        "mapping:m_ORDERS_STG_TO_FACT_ORDERS#overview",
        "mapping:m_ORDERS_STG_TO_FACT_ORDERS#transformations",
        "mapping:m_ORDERS_STG_TO_FACT_ORDERS#field_mappings",
    }
    assert "$$LAST_EXTRACT_TS" in by_loc["mapping:m_ORDERS_STG_TO_FACT_ORDERS#overview"].text
    fields = by_loc["mapping:m_ORDERS_STG_TO_FACT_ORDERS#field_mappings"].text
    assert "ORDER_TOTAL -> GROSS_AMOUNT" in fields


def test_markdown_sections_have_line_ranges_and_skip_banner():
    doc = _parse("runbooks/incremental_load_fact_orders.md")
    locs = [c.locator for c in doc.chunks]
    assert "section:Rerunning a failed run" in locs
    assert not any("(intro)" in loc for loc in locs), "blockquote-only banner must not be indexed"
    rerun = next(c for c in doc.chunks if c.locator == "section:Rerunning a failed run")
    source = (CORPUS / "runbooks/incremental_load_fact_orders.md").read_text().splitlines()
    assert source[rerun.start_line - 1] == "## Rerunning a failed run"
    assert rerun.end_line > rerun.start_line


def test_markdown_heading_inside_code_fence_is_not_a_section():
    raw = b"# T\n\n## Real\n\ntext here\n\n```sql\n# not a heading\nSELECT 1;\n```\n"
    doc = parse_document("x.md", raw)
    assert [c.locator for c in doc.chunks] == ["section:Real"]
    assert "# not a heading" in doc.chunks[0].text


def test_chunk_ids_are_deterministic_and_independent_of_text():
    a = build_chunks(_parse("runbooks/failed_etl_job_triage.md"))
    b = build_chunks(_parse("runbooks/failed_etl_job_triage.md"))
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert make_chunk_id("d.md", "section:A") == make_chunk_id("d.md", "section:A")
    assert make_chunk_id("d.md", "section:A") != make_chunk_id("e.md", "section:A")


@pytest.mark.parametrize(
    ("name", "raw", "message"),
    [
        ("bad.json", b"{not json", "not valid UTF-8 JSON"),
        ("nofmt.json", b'{"tables": []}', "unknown or missing 'format'"),
        ("bad.md", b"no title here\n\nbody", "must start with a '# Title'"),
        ("empty.md", b"   \n", "empty document"),
        ("x.csv", b"a,b\n1,2", "unsupported file type"),
        ("bad.xml", b"<MAPPING_EXPORT", "malformed XML"),
        ("wrong.xml", b"<ROOT/>", "root must be"),
    ],
)
def test_invalid_documents_are_rejected(name, raw, message):
    with pytest.raises(DocumentValidationError, match=message):
        parse_document(name, raw)


def test_snowflake_schema_validation_errors_are_readable():
    data = json.loads((CORPUS / "snowflake/raw_tables.json").read_text())
    data["tables"][0]["columns"][0]["name"] = "order id"  # invalid identifier
    with pytest.raises(DocumentValidationError, match="invalid column identifier"):
        parse_document("t.json", json.dumps(data).encode())

    data = json.loads((CORPUS / "snowflake/raw_tables.json").read_text())
    data["tables"].append(data["tables"][0])
    with pytest.raises(DocumentValidationError, match="duplicate table RAW.ORDERS_STG"):
        parse_document("t.json", json.dumps(data).encode())


def test_mapping_without_target_is_rejected():
    raw = b"""<MAPPING_EXPORT FORMAT="dpka.informatica_mapping.v1">
      <MAPPING NAME="m_X"><SOURCE NAME="A"/>
      <FIELD_MAPPING SOURCE_FIELD="a" TARGET_FIELD="b"/></MAPPING></MAPPING_EXPORT>"""
    with pytest.raises(DocumentValidationError, match="needs at least one SOURCE and one TARGET"):
        parse_document("m.xml", raw)


def test_xml_entity_expansion_is_refused():
    raw = b"""<?xml version="1.0"?>
<!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;&lol;">]>
<MAPPING_EXPORT FORMAT="dpka.informatica_mapping.v1">&lol2;</MAPPING_EXPORT>"""
    with pytest.raises(DocumentValidationError, match="rejected XML"):
        parse_document("evil.xml", raw)
