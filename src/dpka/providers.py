"""Answer providers.

* ``ExtractiveProvider`` - OFFLINE RETRIEVAL DEMONSTRATION. No language model: it selects the
  evidence sentences that best overlap the question and cites them. Free, deterministic, runs
  anywhere. It cannot paraphrase, combine facts or reason.
* ``BedrockProvider`` - generation through the AWS Bedrock Converse API (any Converse-capable
  model enabled in your account). Requires AWS credentials and incurs Bedrock charges.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from dpka.prompting import INSUFFICIENT, SYSTEM_PROMPT, Evidence, build_user_prompt
from dpka.safety import is_suspicious
from dpka.text import content_terms


class ProviderError(RuntimeError):
    """Generation failed. Retrieval results are still returned to the caller."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class AnswerProvider(Protocol):
    name: str
    label: str

    def generate_answer(self, question: str, evidence: list[Evidence]) -> str: ...


_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`(])")


_ITEM = re.compile(r"^(?:[-*]\s+|\d+\.\s+)")
_TABLE_SEP = re.compile(r"^\|?[\s:|-]+\|?$")


def _paragraphs(text: str) -> list[list[str]]:
    """Blank-line separated blocks, without the breadcrumb line and fenced code."""
    paras: list[list[str]] = [[]]
    in_code = False
    for line in text.splitlines()[1:]:  # first line is the breadcrumb/header
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if not s:
            paras.append([])
        else:
            paras[-1].append(s)
    return [p for p in paras if p]


def _units(para: list[str], prose: bool) -> list[str]:
    """Split a paragraph into list items / table rows; re-join wrapped lines if ``prose``."""
    units: list[str] = []
    for i, line in enumerate(para):
        if line.startswith("|"):
            nxt = para[i + 1] if i + 1 < len(para) else ""
            if _TABLE_SEP.match(line) or _TABLE_SEP.match(nxt):
                continue  # separator or header row
            units.append(line)
        elif not prose or _ITEM.match(line) or not units or units[-1].startswith("|"):
            units.append(_ITEM.sub("", line))
        else:
            units[-1] += " " + line  # continuation of a wrapped line
    return units


def _sentences(para: list[str], prose: bool) -> list[str]:
    out = []
    for unit in _units(para, prose):
        out.extend(p.strip() for p in _SPLIT.split(unit) if len(p.strip()) > 12)
    return out


class ExtractiveProvider:
    name = "offline-extractive"
    label = "Offline retrieval demonstration (extractive, no LLM)"

    def __init__(self, max_sentences: int = 3, min_overlap: int = 2):
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    def generate_answer(self, question: str, evidence: list[Evidence]) -> str:
        q_terms = content_terms(question)
        if not q_terms:
            return INSUFFICIENT
        need = min(self.min_overlap, len(q_terms))
        scored: list[tuple[float, int, str, str]] = []
        for rank, e in enumerate(evidence):
            sentences = []
            for para in _paragraphs(e.chunk.text):
                # Drop the whole paragraph around injected instructions, not just the
                # sentence that matched a pattern: the rest of it is part of the payload.
                if not is_suspicious(" ".join(para)):
                    # runbooks are wrapped Markdown prose; generated chunks are one fact per line
                    sentences.extend(_sentences(para, prose=e.chunk.doc_type == "runbook"))
            for j, sent in enumerate(sentences):
                overlap = len(q_terms & content_terms(sent))
                if overlap >= need:
                    # more matched terms first, then better-ranked evidence, then document order
                    scored.append((overlap - 0.1 * rank - 0.001 * j, rank, sent, e.label))
        if not scored:
            return INSUFFICIENT
        scored.sort(key=lambda s: -s[0])
        picked, seen = [], set()
        for _, _, sent, label in scored:
            if sent in seen:
                continue
            seen.add(sent)
            picked.append(f"- {sent} [{label}]")
            if len(picked) == self.max_sentences:
                break
        return "\n".join(picked)


class BedrockProvider:
    label = "AWS Bedrock (Converse API)"

    def __init__(
        self,
        model_id: str,
        region: str,
        *,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        max_tokens: int = 1024,
        temperature: float | None = None,
        client: Any | None = None,
    ):
        # model_id may be a base model ID, an inference profile ID or ARN, or a provisioned /
        # custom model ARN: Converse accepts all of them in the same field.
        if not model_id:
            raise ValueError("Bedrock model id is required")
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.name = f"bedrock:{model_id}"
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=Config(
                    connect_timeout=5,
                    read_timeout=timeout_s,
                    retries={"max_attempts": max_retries + 1, "mode": "standard"},
                ),
            )
        self._client = client

    def generate_answer(self, question: str, evidence: list[Evidence]) -> str:
        from botocore.exceptions import BotoCoreError, ClientError

        # Sampling parameters are only sent when configured: supported values differ between
        # models, so the default is to leave them to the model.
        inference: dict[str, Any] = {"maxTokens": self.max_tokens}
        if self.temperature is not None:
            inference["temperature"] = self.temperature
        try:
            resp = self._client.converse(
                modelId=self.model_id,
                system=[{"text": SYSTEM_PROMPT}],
                messages=[
                    {"role": "user", "content": [{"text": build_user_prompt(question, evidence)}]}
                ],
                inferenceConfig=inference,
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "ClientError")
            retryable = code in {
                "ThrottlingException",
                "ServiceUnavailableException",
                "InternalServerException",
                "ModelNotReadyException",
                "ModelTimeoutException",
            }
            raise ProviderError(f"Bedrock request failed: {code}", retryable=retryable) from exc
        except BotoCoreError as exc:
            # credentials missing, endpoint unreachable, read timeout, ...
            raise ProviderError(
                f"Bedrock unavailable: {exc.__class__.__name__}", retryable=True
            ) from exc

        stop = resp.get("stopReason")
        if stop in {"guardrail_intervened", "content_filtered"}:
            raise ProviderError(f"Bedrock blocked the response (stopReason={stop})")
        if stop in {"malformed_model_output", "model_context_window_exceeded"}:
            raise ProviderError(f"Bedrock could not complete the answer (stopReason={stop})")
        try:
            parts = resp["output"]["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ProviderError("Bedrock returned an unexpected response shape") from exc
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
        if not text:
            raise ProviderError(f"Bedrock returned no text (stopReason={stop})")
        if stop == "max_tokens":
            text += "\n\n(Answer truncated at the token limit.)"
        return text


def build_provider(settings: Any) -> AnswerProvider:
    if settings.llm_provider == "offline":
        return ExtractiveProvider()
    if settings.llm_provider == "bedrock":
        return BedrockProvider(
            settings.bedrock_model_id,
            settings.aws_region,
            timeout_s=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
    raise ValueError(f"unknown provider {settings.llm_provider!r}")
