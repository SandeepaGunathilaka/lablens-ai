"""Checks that keep generated text educational and faithful to the request."""

import json
import re

from pydantic import BaseModel, Field, ValidationError

from explanation_agent.copy import required_explanation_sentences
from explanation_agent.models import ExplanationTask

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_DISALLOWED = (
    ("you have", re.compile(r"\byou have\b(?!\s+not\b)", re.IGNORECASE)),
    ("you've got", re.compile(r"\byou've got\b", re.IGNORECASE)),
    ("you suffer", re.compile(r"\byou suffer\b", re.IGNORECASE)),
    ("you are suffering", re.compile(r"\byou(?: are|'re) suffering\b", re.IGNORECASE)),
    ("diagnosis language", re.compile(r"\bdiagnos\w*\b", re.IGNORECASE)),
    ("prescription language", re.compile(r"\bprescri\w*\b", re.IGNORECASE)),
    ("dosage", re.compile(r"\bdosage\b|\bdose of\b", re.IGNORECASE)),
    ("you should take", re.compile(r"\byou should take\b", re.IGNORECASE)),
    ("you need to take", re.compile(r"\byou need to take\b", re.IGNORECASE)),
    ("start taking", re.compile(r"\bstart taking\b", re.IGNORECASE)),
    ("stop taking", re.compile(r"\bstop taking\b", re.IGNORECASE)),
    ("treatment is", re.compile(r"\btreatment is\b", re.IGNORECASE)),
)

_MAX_FIELD_LENGTH = 1500


class ModelDraft(BaseModel):
    what_it_measures: str = ""
    explanation: str = ""
    possible_meaning: str = ""
    recommended_discussion: str = ""
    insufficient_information: bool = False
    sources_used: list[str] = Field(default_factory=list)


class DraftParseError(ValueError):
    """The model did not return the required JSON object."""


def parse_model_draft(raw: str) -> ModelDraft:
    text = raw.strip()
    if text.startswith("```"):
        text = _FENCE.sub("", text).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DraftParseError("model output was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise DraftParseError("model output was not a JSON object")
    try:
        draft = ModelDraft.model_validate(payload)
    except ValidationError as exc:
        raise DraftParseError("model output did not match the required JSON fields") from exc
    draft.sources_used = [item.strip() for item in draft.sources_used if item.strip()]
    return draft


def disallowed_reasons(text: str) -> list[str]:
    reasons = []
    for label, pattern in _DISALLOWED:
        if pattern.search(text):
            reasons.append(f"disallowed language: {label}")
    return reasons


def validate_model_draft(draft: ModelDraft, task: ExplanationTask) -> list[str]:
    """Return rejection reasons. An empty list means the draft can be shown."""

    if draft.insufficient_information:
        return []

    reasons: list[str] = []
    fields = {
        "what_it_measures": draft.what_it_measures.strip(),
        "explanation": draft.explanation.strip(),
        "possible_meaning": draft.possible_meaning.strip(),
        "recommended_discussion": draft.recommended_discussion.strip(),
    }
    for name, text in fields.items():
        if len(text) < 20:
            reasons.append(f"{name} is too short")
        if len(text) > _MAX_FIELD_LENGTH:
            reasons.append(f"{name} exceeds the length limit")

    explanation = fields["explanation"]
    for sentence in required_explanation_sentences(task):
        if sentence not in explanation:
            reasons.append(f"explanation is missing this exact sentence: {sentence}")

    if "qualified healthcare professional" not in fields["recommended_discussion"]:
        reasons.append(
            "recommended discussion does not direct the reader to a qualified healthcare professional"
        )

    narrative = "\n".join(fields.values())
    reasons.extend(disallowed_reasons(narrative))
    reasons.extend(disallowed_reasons("\n".join(draft.sources_used)))

    allowed = {passage.title.casefold() for passage in task.passages}
    if not draft.sources_used:
        reasons.append("sources_used is empty")
    for title in draft.sources_used:
        if title.casefold() not in allowed:
            reasons.append(f"sources_used includes a title that was not retrieved: {title}")
        if title not in narrative:
            reasons.append(f"the draft does not mention the retrieved source: {title}")

    return reasons
