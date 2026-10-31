from __future__ import annotations

from dpka.citations import check_citations

EVIDENCE = {
    "S1": "The watermark LAST_EXTRACT_TS is stored in AUDIT.ETL_CONTROL and advanced on success.",
    "S2": "DIM_PRODUCT is rebuilt nightly with a truncate and insert in one transaction.",
}


def test_valid_citations_are_kept_in_order_of_use():
    c = check_citations(
        "DIM_PRODUCT is rebuilt nightly in one transaction [S2]. "
        "The watermark is stored in AUDIT.ETL_CONTROL [S1].",
        EVIDENCE,
    )
    assert c.valid_labels == ["S2", "S1"]
    assert c.invalid_labels == [] and c.warnings == []


def test_unknown_labels_are_removed_and_reported():
    c = check_citations(
        "The watermark is stored in AUDIT.ETL_CONTROL [S1, S7]. More [S9].", EVIDENCE
    )
    assert c.valid_labels == ["S1"]
    assert c.invalid_labels == ["S7", "S9"]
    assert "[S1]" in c.text and "S7" not in c.text and "S9" not in c.text
    assert any("unknown sources" in w for w in c.warnings)


def test_citation_after_period_stays_with_its_sentence():
    c = check_citations("The watermark is stored in AUDIT.ETL_CONTROL. [S1]", EVIDENCE)
    assert c.uncited_sentences == []


def test_uncited_and_weakly_supported_sentences_are_flagged():
    c = check_citations(
        "The watermark is stored in AUDIT.ETL_CONTROL [S1]. "
        "Kafka consumers replicate inventory snapshots hourly. "
        "Quarterly revenue dashboards refresh from Tableau extracts [S2].",
        EVIDENCE,
    )
    assert any("Kafka" in s for s in c.uncited_sentences)
    assert any("Tableau" in s for s in c.weakly_supported)
    assert len(c.warnings) == 2


def test_no_citations_at_all():
    c = check_citations("Everything is fine, rerun the job.", EVIDENCE)
    assert c.valid_labels == []
