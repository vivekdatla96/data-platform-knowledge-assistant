"""Bedrock provider tests using botocore's Stubber: request/response shapes, no network calls."""

from __future__ import annotations

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.stub import ANY, Stubber  # noqa: E402

from dpka.assistant import Assistant  # noqa: E402
from dpka.config import Settings  # noqa: E402
from dpka.prompting import SYSTEM_PROMPT, Evidence  # noqa: E402
from dpka.providers import BedrockProvider, ProviderError  # noqa: E402

MODEL = "example.test-model-v1"


@pytest.fixture
def client():
    c = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    with Stubber(c) as stub:
        c.stub = stub  # type: ignore[attr-defined]
        yield c


def _ok(text: str, stop: str = "end_turn") -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": stop,
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        "metrics": {"latencyMs": 1},
    }


def _evidence(retriever) -> list[Evidence]:
    return [
        Evidence(f"S{i}", h.chunk)
        for i, h in enumerate(retriever.search("backfill FACT_ORDERS", k=2), start=1)
    ]


def test_converse_request_shape_and_answer(client, retriever):
    expected = {
        "modelId": MODEL,
        "system": [{"text": SYSTEM_PROMPT}],
        "messages": ANY,
        "inferenceConfig": {"maxTokens": 1024},  # no sampling parameters
    }
    client.stub.add_response("converse", _ok("Set OVERRIDE_START_TS [S1]."), expected)
    p = BedrockProvider(MODEL, "us-east-1", client=client)
    assert (
        p.generate_answer("How do I backfill?", _evidence(retriever))
        == "Set OVERRIDE_START_TS [S1]."
    )
    client.stub.assert_no_pending_responses()


def test_user_message_contains_delimited_evidence(client, retriever):
    captured = {}

    def check(params):
        captured.update(params)

    client.meta.events.register(
        "provide-client-params.bedrock-runtime.Converse", lambda params, **_: check(params)
    )
    client.stub.add_response("converse", _ok("x [S1]"), None)
    BedrockProvider(MODEL, "us-east-1", client=client).generate_answer("q?", _evidence(retriever))
    user_text = captured["messages"][0]["content"][0]["text"]
    assert user_text.startswith('<evidence id="S1"')
    assert user_text.rstrip().endswith("Question: q?")


@pytest.mark.parametrize(
    ("code", "retryable"),
    [
        ("ThrottlingException", True),
        ("ServiceUnavailableException", True),
        ("AccessDeniedException", False),
        ("ValidationException", False),
    ],
)
def test_client_errors_become_provider_errors(client, retriever, code, retryable):
    client.stub.add_client_error("converse", service_error_code=code, http_status_code=400)
    p = BedrockProvider(MODEL, "us-east-1", client=client)
    with pytest.raises(ProviderError) as ei:
        p.generate_answer("q", _evidence(retriever))
    assert code in str(ei.value)
    assert ei.value.retryable is retryable


def test_guardrail_and_empty_responses_are_errors(client, retriever):
    p = BedrockProvider(MODEL, "us-east-1", client=client)
    client.stub.add_response("converse", _ok("blocked", stop="guardrail_intervened"))
    with pytest.raises(ProviderError, match="guardrail_intervened"):
        p.generate_answer("q", _evidence(retriever))
    client.stub.add_response("converse", _ok("   "))
    with pytest.raises(ProviderError, match="no text"):
        p.generate_answer("q", _evidence(retriever))


def test_truncated_answer_is_marked(client, retriever):
    client.stub.add_response("converse", _ok("Partial answer [S1]", stop="max_tokens"))
    out = BedrockProvider(MODEL, "us-east-1", client=client).generate_answer(
        "q", _evidence(retriever)
    )
    assert "truncated" in out


def test_bedrock_failure_end_to_end_degrades_to_evidence(client, retriever):
    client.stub.add_client_error("converse", service_error_code="ThrottlingException")
    a = Assistant(
        retriever,
        BedrockProvider(MODEL, "us-east-1", client=client),
        min_relevance=0.0,
        min_coverage=0.3,
    )
    res = a.ask("How do I backfill FACT_ORDERS?")
    assert res.status == "generation_failed" and res.evidence


def test_bedrock_answer_end_to_end_is_citation_checked(client, retriever):
    client.stub.add_response(
        "converse",
        _ok(
            "Set OVERRIDE_START_TS in AUDIT.ETL_CONTROL for the backfill range [S1]. "
            "Never truncate DW.FACT_ORDERS [S1]."
        ),
    )
    a = Assistant(
        retriever,
        BedrockProvider(MODEL, "us-east-1", client=client),
        min_relevance=0.0,
        min_coverage=0.3,
    )
    res = a.ask("How do I backfill FACT_ORDERS for a past date range?")
    assert res.status == "answered"
    assert res.provider == f"bedrock:{MODEL}"
    assert res.citations[0]["label"] == "S1"


def test_settings_require_model_id_for_bedrock():
    with pytest.raises(ValueError, match="DPKA_BEDROCK_MODEL_ID"):
        Settings(llm_provider="bedrock")
    s = Settings.from_env(
        {"DPKA_LLM_PROVIDER": "bedrock", "DPKA_BEDROCK_MODEL_ID": MODEL, "DPKA_TOP_K": "7"}
    )
    assert s.bedrock_model_id == MODEL and s.top_k == 7


def test_inference_profile_arn_and_optional_temperature(client, retriever):
    arn = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.example.test-model-v1"
    client.stub.add_response(
        "converse",
        _ok("x [S1]"),
        {
            "modelId": arn,
            "system": ANY,
            "messages": ANY,
            "inferenceConfig": {"maxTokens": 256, "temperature": 0.0},
        },
    )
    p = BedrockProvider(arn, "us-east-1", client=client, max_tokens=256, temperature=0.0)
    assert p.generate_answer("q", _evidence(retriever)) == "x [S1]"
    s = Settings.from_env(
        {
            "DPKA_LLM_PROVIDER": "bedrock",
            "DPKA_BEDROCK_MODEL_ID": arn,
            "DPKA_LLM_TEMPERATURE": "0.2",
        }
    )
    assert s.llm_temperature == 0.2
    assert Settings().llm_temperature is None


def test_model_timeout_is_retryable_and_bad_stop_reasons_fail(client, retriever):
    p = BedrockProvider(MODEL, "us-east-1", client=client)
    client.stub.add_client_error(
        "converse", service_error_code="ModelTimeoutException", http_status_code=408
    )
    with pytest.raises(ProviderError) as ei:
        p.generate_answer("q", _evidence(retriever))
    assert ei.value.retryable
    client.stub.add_response("converse", _ok("partial", stop="model_context_window_exceeded"))
    with pytest.raises(ProviderError, match="model_context_window_exceeded"):
        p.generate_answer("q", _evidence(retriever))
