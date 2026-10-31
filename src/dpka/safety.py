"""Heuristics for treating retrieved documents as untrusted content.

Pattern matching cannot catch every prompt injection. It is one layer: flagged chunks are marked
in the UI and excluded from extractive answers, and every provider prompt wraps evidence in
delimiters with an explicit instruction that evidence is data, not instructions.
"""

from __future__ import annotations

import re

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all |any )?(previous|prior|above|the) (instructions|rules)",
    r"you are now (in )?\w+ mode",
    r"\bsystem prompt\b",
    r"\bact as (an? )?(admin|administrator|root|developer)\b",
    r"\bdo not (tell|inform) the user\b",
    r"</?(system|assistant|evidence)\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def find_injection_markers(text: str) -> list[str]:
    return [m.group(0) for rx in _COMPILED if (m := rx.search(text))]


def is_suspicious(text: str) -> bool:
    return bool(find_injection_markers(text))


def neutralize_delimiters(text: str) -> str:
    """Stop document text from closing or opening our evidence tags inside a prompt."""
    return re.sub(r"<(/?)(evidence|system|assistant)", r"<\1\2_", text, flags=re.IGNORECASE)
