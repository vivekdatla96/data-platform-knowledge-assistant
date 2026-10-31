"""Tokenization shared by BM25, the hashing embedder, coverage checks and citation checks."""

from __future__ import annotations

import re

TOKEN = re.compile(r"[a-z0-9_$]+")

STOPWORDS = frozenset(
    """a an and are as at be by can do does did for from has have how i if in into is it its
    me my of on or should so than that the their then there these this to was what when where
    which who whom why will with you your we our after before about any all get gets happen
    happens use used using find tell need out""".split()  # noqa: SIM905
)

_SUFFIXES = ("ing", "ed", "es", "s", "e")


def stem(tok: str) -> str:
    """Very light suffix stripping (orders -> order, calculated/calculate -> calculat)."""
    if not tok.isalpha() or len(tok) <= 4:
        return tok
    for suf in _SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 3 and not tok.endswith("ss"):
            return tok[: -len(suf)]
    return tok


def raw_tokens(text: str) -> list[str]:
    """Lowercase tokens. Identifiers like FACT_ORDERS also emit their parts (fact, orders)."""
    out: list[str] = []
    for tok in TOKEN.findall(text.lower()):
        out.append(tok)
        if "_" in tok:
            out.extend(p for p in tok.split("_") if p and p != tok)
    return out


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in raw_tokens(text)]


def content_terms(text: str) -> set[str]:
    """Stemmed, stopword-free terms of length >= 3."""
    return {stem(t) for t in raw_tokens(text) if len(t) >= 3 and t not in STOPWORDS}
