"""FastAPI app: a JSON API and a single-page UI showing question, answer and evidence."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from dpka.assistant import MAX_QUESTION_CHARS, Assistant
from dpka.config import Settings

SAMPLE_QUESTIONS = [
    "How does the FACT_ORDERS incremental load decide which source rows to extract?",
    "What should I do when an ETL session fails with a statement timeout?",
    "How is NET_AMOUNT calculated in FACT_ORDERS?",
    "Which downstream objects depend on DW.FACT_ORDERS?",
    "How are partner return files loaded into RAW.RETURNS_STG?",
    "What is the Snowflake account URL for production?",
]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int | None = Field(default=None, ge=1, le=20)


def create_app(settings: Settings | None = None, assistant: Assistant | None = None) -> FastAPI:
    app = FastAPI(
        title="Data Platform Knowledge Assistant",
        description="Portfolio demo over synthetic data. Answers cite retrieved sources.",
        version="0.1.0",
    )
    state: dict[str, Assistant] = {}

    def get_assistant() -> Assistant:
        if "a" not in state:
            state["a"] = assistant or Assistant.from_settings(settings or Settings.from_env())
        return state["a"]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return files("dpka").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        a = get_assistant()
        return {
            "status": "ok",
            "index": a.retriever.store.stats(),
            "retrieval_mode": a.retriever.mode,
            "provider": a.provider.name,
            "provider_label": a.provider.label,
            "sample_questions": SAMPLE_QUESTIONS,
        }

    @app.post("/api/ask")
    def ask(req: AskRequest) -> dict[str, Any]:
        try:
            return get_assistant().ask(req.question, top_k=req.top_k).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
