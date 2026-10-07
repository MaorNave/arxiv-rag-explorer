"""Factories for the local chat model (Qwen via Ollama) and the query analyzer."""

from __future__ import annotations

from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from .config import Settings

THINKING_BUDGET = 3072  # extra tokens for the model's reasoning when LLM_THINKING=true


class QueryPlan(BaseModel):
    """Search plan produced by the "understand query" step."""

    search_queries: list[str] = Field(
        default_factory=list, max_length=2,
        description="1-2 short (max 12 words), self-contained English search queries",
    )
    keywords: list[str] = Field(
        default_factory=list, max_length=5,
        description="Up to 5 distinctive technical terms, names or acronyms (English)",
    )
    topic: str = Field(default="", description="3-6 word English noun phrase naming the subject")


def build_chat_model(
    settings: Settings,
    *,
    temperature: float | None = None,
    num_predict: int | None = None,
    thinking: bool | None = None,
    supports_thinking: bool = True,
) -> ChatOllama:
    # reasoning=False sends think=false (fast, direct answers); True streams the
    # model's reasoning separately; None leaves models without a thinking mode alone.
    if not supports_thinking:
        reasoning = None
    else:
        reasoning = settings.llm_thinking if thinking is None else thinking
    num_predict = settings.llm_num_predict if num_predict is None else num_predict
    if reasoning:  # Ollama counts reasoning tokens against num_predict: reserve room for both
        num_predict += THINKING_BUDGET
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature if temperature is None else temperature,
        num_ctx=settings.llm_num_ctx,
        num_predict=num_predict,
        reasoning=reasoning,
        keep_alive=settings.llm_keep_alive,
        client_kwargs={"timeout": settings.llm_timeout_s},
    )


def build_analyzer(settings: Settings, *, supports_thinking: bool = True) -> Runnable:
    """Chat model constrained (via Ollama JSON-schema output) to return a QueryPlan."""
    model = build_chat_model(
        settings, temperature=0.0, num_predict=300, thinking=False, supports_thinking=supports_thinking
    )
    return model.with_structured_output(QueryPlan, method="json_schema")
