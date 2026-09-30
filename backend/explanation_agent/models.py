"""Structured input and output for the Explanation Agent.

The request uses the pipeline's shared shapes: findings carry the Document Agent's
``test`` / numeric ``value`` fields plus the status the Coordinator computed in code,
and ``retrieved_sources`` is the same list the Coordinator sends to the Safety Agent.
"""

import math
from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from agents.safety_agent import RetrievedSource

Status = Literal["low", "normal", "high"]
GenerationMode = Literal[
    "llm",
    "template",
    "insufficient",
    "safe_fallback",
    "unavailable",
]

RequestId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")]


class ExplanationFinding(BaseModel):
    """One extracted lab value. ``status`` is None when the Coordinator could not compute it."""

    test: str = Field(min_length=1, max_length=200)
    value: float
    unit: str | None = Field(default=None, max_length=40)
    reference_range: str | None = Field(default=None, max_length=200)
    status: Status | None = None

    @field_validator("test", mode="before")
    @classmethod
    def strip_test(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("unit", "reference_range", mode="before")
    @classmethod
    def blank_optional_to_none(cls, value: object) -> object:
        if value is None or not isinstance(value, str):
            return value
        stripped = value.strip()
        return stripped or None

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be a finite number")
        return value


class ExplanationRequest(BaseModel):
    task_id: RequestId
    report_id: RequestId
    user_id: RequestId
    findings: list[ExplanationFinding] = Field(min_length=1, max_length=50)
    retrieved_sources: list[RetrievedSource] = Field(default_factory=list, max_length=50)
    rejection_feedback: list[str] = Field(default_factory=list, max_length=8)
    user_question: str | None = Field(default=None, max_length=1000)

    @field_validator("user_question", mode="before")
    @classmethod
    def strip_question(cls, value: object) -> object:
        if value is None or not isinstance(value, str):
            return value
        stripped = value.strip()
        return stripped or None

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


class ExplainedFinding(BaseModel):
    test_name: str
    result: str
    status: Status | None
    what_it_measures: str
    explanation: str
    possible_meaning: str
    recommended_discussion: str
    insufficient_information: bool
    sources_used: list[str]
    generation_mode: GenerationMode


class ExplanationResponse(BaseModel):
    task_id: str
    report_id: str
    user_id: str
    findings: list[ExplainedFinding]
    # For the Coordinator and audit trail only; never shown to the patient.
    safety_notes: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class Passage:
    """One cited piece of retrieved text the model may use."""

    title: str
    url: str | None
    excerpt: str


@dataclass(frozen=True)
class ExplanationTask:
    """Everything needed to explain a single finding."""

    test_name: str
    value: float
    unit: str | None
    reference_range: str | None
    status: Status | None
    passages: list[Passage] = field(default_factory=list)
    rejection_feedback: list[str] = field(default_factory=list)
    user_question: str | None = None
