"""Citation parsing and verification for generated answers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from dpka.text import content_terms

CITATION = re.compile(r"\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]")
# split after sentence punctuation, but keep "... sentence. [S1]" together
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?!\[)|\n+")
MIN_SUPPORT = 0.4  # share of a sentence's content words that must appear in its cited evidence


@dataclass
class CitationCheck:
    text: str  # answer with unknown labels removed
    valid_labels: list[str] = field(default_factory=list)  # in order of first use
    invalid_labels: list[str] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    weakly_supported: list[str] = field(default_factory=list)

    @property
    def warnings(self) -> list[str]:
        w = []
        if self.invalid_labels:
            w.append(f"Removed citations to unknown sources: {', '.join(self.invalid_labels)}")
        for s in self.uncited_sentences:
            w.append(f"Sentence without citation: {s[:120]}")
        for s in self.weakly_supported:
            w.append(f"Low word overlap with cited evidence (check manually): {s[:120]}")
        return w


def check_citations(answer: str, evidence_text: dict[str, str]) -> CitationCheck:
    """Validate ``[S#]`` citations against the evidence actually given to the provider.

    ``evidence_text`` maps label -> chunk text. Unknown labels are stripped from the answer and
    reported; sentences are checked for a citation and for lexical support in what they cite.
    """
    check = CitationCheck(text=answer)

    def _rewrite(m: re.Match[str]) -> str:
        labels = [x.strip() for x in m.group(1).split(",")]
        keep = []
        for lab in labels:
            if lab in evidence_text:
                keep.append(lab)
                if lab not in check.valid_labels:
                    check.valid_labels.append(lab)
            elif lab not in check.invalid_labels:
                check.invalid_labels.append(lab)
        return f"[{', '.join(keep)}]" if keep else ""

    check.text = CITATION.sub(_rewrite, answer).strip()

    for sent in SENTENCE_SPLIT.split(check.text):
        s = sent.strip().lstrip("-* ").strip()
        terms = content_terms(CITATION.sub("", s))
        if len(terms) < 3:
            continue  # headings, fragments, "(Answer truncated...)"
        cited = [x.strip() for m in CITATION.finditer(s) for x in m.group(1).split(",")]
        if not cited:
            check.uncited_sentences.append(s)
            continue
        support = set().union(*(content_terms(evidence_text[c]) for c in cited))
        if len(terms & support) / len(terms) < MIN_SUPPORT:
            check.weakly_supported.append(s)
    return check
