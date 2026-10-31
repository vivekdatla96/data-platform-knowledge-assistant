"""Runtime settings, read from environment variables (prefix ``DPKA_``).

Every setting has a default that runs the fully offline demo, so nothing is required to start.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

EMBEDDING_BACKENDS = ("fastembed", "hashing")
RETRIEVAL_MODES = ("dense", "hybrid")
LLM_PROVIDERS = ("offline", "bedrock")


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path = Path("data/corpus")
    index_path: Path = Path(".dpka/index.sqlite")

    # "fastembed" runs an open-source ONNX model locally. "hashing" is a deterministic,
    # dependency-free embedder used by the unit tests and CI; it is not a semantic model.
    embedding_backend: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    retrieval_mode: str = "hybrid"
    top_k: int = 5
    # Abstention gate (both must pass): best dense cosine similarity among retrieved chunks,
    # and IDF-weighted share of question terms found in the retrieved text. Chosen on the
    # synthetic eval set for bge-small-en-v1.5; see docs/EVALUATION.md.
    min_relevance: float = 0.68
    min_coverage: float = 0.3

    llm_provider: str = "offline"
    bedrock_model_id: str = ""
    aws_region: str = "us-east-1"
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    llm_max_tokens: int = 1024
    # Optional; omitted from the request when unset. Check your model's supported parameters.
    llm_temperature: float | None = None

    def __post_init__(self) -> None:
        if self.embedding_backend not in EMBEDDING_BACKENDS:
            raise ValueError(f"DPKA_EMBEDDING_BACKEND must be one of {EMBEDDING_BACKENDS}")
        if self.retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"DPKA_RETRIEVAL_MODE must be one of {RETRIEVAL_MODES}")
        if self.llm_provider not in LLM_PROVIDERS:
            raise ValueError(f"DPKA_LLM_PROVIDER must be one of {LLM_PROVIDERS}")
        if not 1 <= self.top_k <= 20:
            raise ValueError("DPKA_TOP_K must be between 1 and 20")
        if self.llm_provider == "bedrock" and not self.bedrock_model_id:
            raise ValueError("DPKA_BEDROCK_MODEL_ID is required when DPKA_LLM_PROVIDER=bedrock")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides: object) -> Settings:
        env = os.environ if env is None else env
        values: dict[str, object] = {}
        for f in fields(cls):
            raw = env.get(f"DPKA_{f.name.upper()}")
            if raw is None or raw == "":
                continue
            default = f.default
            if default is None:  # only optional float settings use None
                values[f.name] = float(raw)
            elif isinstance(default, Path):
                values[f.name] = Path(raw)
            elif isinstance(default, bool):
                values[f.name] = raw.lower() in {"1", "true", "yes"}
            elif isinstance(default, int):
                values[f.name] = int(raw)
            elif isinstance(default, float):
                values[f.name] = float(raw)
            else:
                values[f.name] = raw
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)  # type: ignore[arg-type]
