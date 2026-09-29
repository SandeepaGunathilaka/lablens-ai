"""Structured input and output for the Explanation Agent."""

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Status = Literal["low", "normal", "high", "unknown"]
GenerationMode = Literal[
    "llm",
    "template",
    "insufficient",
    "safe_fallback",
    "unavailable",
]


class RetrievedSource(BaseModel):
    """One chunk from the Retrieval Agent. Optional on the request."""

    title: str = Field(min_length=1, max_length=300)
    excerpt: str = Field(min_length=1, max_length=4000)
    url: str | None = Field(default=None, max_length=500)
    source_id: str | None = Field(default=None, max_length=128)

    @field_validator("title", "excerpt", mode="before")
    @classmethod
    def strip_required(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("url", "source_id", mode="before")
    @classmethod
    def blank_optional_to_none(cls, value: object) -> object:
        if value is None or not isinstance(value, str):
            return value
        stripped = value.strip()
        return stripped or None


class ExplanationRequest(BaseModel):
    """Patient result plus optional retrieved context.

    ``status`` is calculated by the Coordinator with ``calculate_status``.
    When it is omitted, the agent calculates it the same way and never asks
    the language model to choose it.
    """

    task_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    test_name: str = Field(min_length=1, max_length=200)
    value: str = Field(min_length=1, max_length=40)
    unit: str = Field(min_length=1, max_length=40)
    reference_range: str | None = Field(default=None, max_length=200)
    status: Status | None = None
    retrieved_sources: list[RetrievedSource] = Field(default_factory=list, max_length=8)
    user_question: str | None = Field(default=None, max_length=1000)
    rejection_feedback: list[str] = Field(default_factory=list, max_length=8)
    previous_draft: str | None = Field(default=None, max_length=8000)

    @field_validator("test_name", "unit", mode="before")
    @classmethod
    def strip_required_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("reference_range", "user_question", "previous_draft", mode="before")
    @classmethod
    def strip_optional_text(cls, value: object) -> object:
        if value is None or not isinstance(value, str):
            return value
        stripped = value.strip()
        return stripped or None

    @field_validator("value", mode="before")
    @classmethod
    def preserve_reported_value(cls, value: object) -> str:
        """Keep report text unchanged. Bare JSON numbers become a stable string."""

        if isinstance(value, bool) or value is None:
            raise ValueError("value must be a number or a numeric string")
        if isinstance(value, (int, float)):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("value must be a finite number")
            return format(number, "g")
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("value must not be empty")
            if len(stripped) > 40:
                raise ValueError("value is too long")
            return stripped
        raise ValueError("value must be a number or a numeric string")

    @field_validator("rejection_feedback")
    @classmethod
    def clean_feedback(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in values:
            text = item.strip()
            if not text:
                continue
            if len(text) > 500:
                raise ValueError("a rejection feedback item is too long")
            cleaned.append(text)
        return cleaned


class ExplanationResponse(BaseModel):
    """Four narrative fields the Coordinator joins into the Safety Agent draft."""

    task_id: str
    test_name: str
    value: str
    unit: str
    reference_range: str | None
    status: Status
    status_detail: str
    what_it_measures: str
    explanation: str
    possible_meaning: str
    recommended_discussion: str
    insufficient_information: bool
    sources_used: list[str]
    generation_mode: GenerationMode
    regenerated: bool
