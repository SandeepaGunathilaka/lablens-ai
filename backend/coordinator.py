"""Coordinator: runs an uploaded lab report through all four agents.

Pipeline: Document -> status (computed here, in code) -> Retrieval -> Explanation
-> Safety, regenerating the explanation up to MAX_EXPLANATION_RETRIES times if Safety
rejects it. Only a Safety-approved draft is ever returned to the caller.

Agents are called as plain Python functions through small, injectable service
interfaces, so tests can pass fakes and unmerged agents can use placeholders.
"""

import logging
import re
import uuid
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field
from pymongo.collection import Collection

from agents.document_agent import DocumentExtractionError, DocumentExtractionResponse, extract_document
from agents.safety_agent import (
    DISCLAIMER,
    ApprovedResponse,
    LabResult,
    RejectedResponse,
    RetrievedSource,
    SafetyValidateRequest,
    validate_draft,
)
from database import get_audit_logs_collection
from logging_service import log_event

logger = logging.getLogger(__name__)

# 2 retries = 3 explanation attempts in total.
MAX_EXPLANATION_RETRIES = 2

INSUFFICIENT_INFORMATION_MESSAGE = (
    "LabLens could not find sufficient information about this test in its approved knowledge base."
)
FALLBACK_MESSAGE = (
    "We couldn't confidently explain this result. Please verify the report with a healthcare professional."
)
EXPLANATION_UNAVAILABLE_MESSAGE = (
    "Explanations are temporarily unavailable. Your extracted lab values are shown below."
)
DOCUMENT_UNAVAILABLE_MESSAGE = "We couldn't read this report right now. Please try again later."
NO_RESULTS_MESSAGE = "No lab values were found in this report."
NEEDS_VERIFICATION_WARNING = (
    "We could not confidently read this value. Please verify it against the original report."
)

# Instruction sent to the Explanation service when Safety rejects a draft, keyed by
# the Safety Agent's rejection reason.
REGENERATION_INSTRUCTIONS = {
    "patient_values_verified": (
        "Regenerate using only the patient's exact values and reference ranges as given; "
        "do not restate or change any number."
    ),
    "diagnosis_detected": (
        "Regenerate without diagnosing: do not say or imply the patient has any condition."
    ),
    "medication_detected": (
        "Regenerate without mentioning any medication, supplement, dose or treatment."
    ),
    "unsupported_claim_detected": (
        "Regenerate using only information from the retrieved sources; do not add any "
        "other medical terms, numbers or claims."
    ),
}
DEFAULT_REGENERATION_INSTRUCTION = "Regenerate the explanation, following all safety rules."

Status = Literal["low", "normal", "high"]


# --- Models ----------------------------------------------------------------------


class Finding(BaseModel):
    """One extracted value plus its code-computed status: the Explanation service's input."""

    test: str
    value: float
    unit: str | None = None
    reference_range: str | None = None
    status: Status | None = None


# MIRRORS Member 3's Explanation Agent models. Replace these two classes with an
# import from the Explanation Agent once that branch is merged.
class ExplainedFinding(BaseModel):
    test_name: str
    result: str
    status: str | None = None
    what_it_measures: str = ""
    explanation: str = ""
    possible_meaning: str = ""
    recommended_discussion: str = ""


class ExplanationOutput(BaseModel):
    findings: list[ExplainedFinding]
    # Safety info from the Explanation Agent. Not shown to the user: only text that
    # passed the Safety Agent is ever returned.
    safety_notes: list[str] = Field(default_factory=list)


class AnalyzedLabResult(BaseModel):
    """An extracted value as returned to the caller: original numbers plus status and warning."""

    test: str
    value: float
    unit: str | None = None
    reference_range: str | None = None
    confidence: float
    needs_verification: bool
    status: Status | None = None
    warning: str | None = None


class CoordinatorResult(BaseModel):
    task_id: str
    report_id: str
    user_id: str
    status: Literal["approved", "fallback", "failed"]
    report_type: str | None = None
    results: list[AnalyzedLabResult] = Field(default_factory=list)
    final_response: str | None = None  # Only ever Safety-approved text.
    message: str | None = None  # Fallback or error text.


# --- Service interfaces ----------------------------------------------------------

# Each service is a plain function; pass your own to analyze_report() to replace it.
DocumentService = Callable[[str, bytes], DocumentExtractionResponse]
RetrievalService = Callable[[list[str]], list[RetrievedSource]]
ExplanationService = Callable[[list[Finding], list[RetrievedSource], str | None], ExplanationOutput]
SafetyService = Callable[[SafetyValidateRequest], ApprovedResponse | RejectedResponse]


def placeholder_retrieval(test_names: list[str]) -> list[RetrievedSource]:
    """PLACEHOLDER until Member 2's Retrieval Agent is merged: finds nothing."""
    return []


def placeholder_explanation(
    findings: list[Finding], retrieved_sources: list[RetrievedSource], instruction: str | None
) -> ExplanationOutput:
    """PLACEHOLDER until Member 3's Explanation Agent is merged.

    Restates only the patient's own values; makes no medical claims.
    """
    explained = []
    for finding in findings:
        value_text = _value_text(finding.value, finding.unit)
        sentence = f"Your {finding.test} value is {value_text}"
        if finding.reference_range:
            sentence += f"; the range on your report is {finding.reference_range}"
        explained.append(
            ExplainedFinding(
                test_name=finding.test,
                result=value_text,
                status=finding.status,
                explanation=sentence + ".",
            )
        )
    return ExplanationOutput(findings=explained)


# --- Status calculation (always in code, never by the LLM) ----------------------

_NUMBER = r"(\d+(?:\.\d+)?)"
_BETWEEN_RE = re.compile(rf"{_NUMBER}\s*(?:-|–|—|to)\s*{_NUMBER}", re.IGNORECASE)
_UPPER_ONLY_RE = re.compile(rf"(<=|<|≤)\s*{_NUMBER}")
_LOWER_ONLY_RE = re.compile(rf"(>=|>|≥)\s*{_NUMBER}")
# "<" and ">" exclude the bound itself; every other form includes it.
_STRICT_OPERATORS = {"<", ">"}


def compute_status(value: float, reference_range: str | None) -> Status | None:
    """Compare a value to its reference range: "low", "normal" or "high".

    Supports "12-16", "12.0 - 16.0", "12 to 16", "<200", "<=200", ">40" and ">=40".
    Two-number ranges and "<="/">=" include their bounds as normal; "<" and ">" are
    strict, so 200 is "high" for "<200" and 40 is "low" for ">40". Returns None when
    the range is missing or doesn't exactly match one of those forms; it never guesses.
    """
    if not reference_range:
        return None
    text = reference_range.strip()

    strict = False
    if match := _BETWEEN_RE.fullmatch(text):
        low, high = float(match.group(1)), float(match.group(2))
        if low > high:
            return None
    elif match := _UPPER_ONLY_RE.fullmatch(text):
        low, high = None, float(match.group(2))
        strict = match.group(1) in _STRICT_OPERATORS
    elif match := _LOWER_ONLY_RE.fullmatch(text):
        low, high = float(match.group(2)), None
        strict = match.group(1) in _STRICT_OPERATORS
    else:
        return None

    if low is not None and (value < low or (strict and value == low)):
        return "low"
    if high is not None and (value > high or (strict and value == high)):
        return "high"
    return "normal"


# --- Draft building --------------------------------------------------------------


def _format_number(value: float) -> str:
    """11.2 -> "11.2", 250.0 -> "250" (never scientific notation)."""
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return text or "0"


def _value_text(value: float, unit: str | None) -> str:
    number = _format_number(value)
    return f"{number} {unit}" if unit else number


def _result_line(finding: Finding) -> str:
    """The value/range/status sentence, built from the Document Agent's numbers and our status."""
    line = f"{finding.test}: your result is {_value_text(finding.value, finding.unit)}"
    if finding.reference_range:
        line += f"; the range on your report is {finding.reference_range}"
    if finding.status:
        line += f", which is {finding.status}"
    return line + "."


def build_draft_response(
    findings: list[Finding],
    explained: list[ExplainedFinding],
    tests_with_sources: set[str],
) -> str:
    """Join every finding into one draft and append the standard DISCLAIMER.

    Result and status always come from code. Only the prose fields come from the
    Explanation service, and only for tests that have retrieved sources; the others
    get INSUFFICIENT_INFORMATION_MESSAGE instead.
    """
    explained_by_test = {}
    for item in explained:
        explained_by_test.setdefault(item.test_name.lower(), item)

    sections = []
    for finding in findings:
        parts = [_result_line(finding)]
        if finding.test.lower() not in tests_with_sources:
            parts.append(INSUFFICIENT_INFORMATION_MESSAGE)
        elif item := explained_by_test.get(finding.test.lower()):
            parts += [
                item.what_it_measures,
                item.explanation,
                item.possible_meaning,
                item.recommended_discussion,
            ]
        sections.append("\n".join(p.strip() for p in parts if p and p.strip()))
    return "\n\n".join(sections + [DISCLAIMER])


# --- Pipeline --------------------------------------------------------------------


def analyze_report(
    filename: str,
    content: bytes,
    user_id: str,
    *,
    document_service: DocumentService | None = None,
    retrieval_service: RetrievalService | None = None,
    explanation_service: ExplanationService | None = None,
    safety_service: SafetyService | None = None,
    audit_logs: Collection | None = None,
) -> CoordinatorResult:
    """Run an uploaded report through the full pipeline. Never raises for agent failures."""
    document_service = document_service or extract_document
    retrieval_service = retrieval_service or placeholder_retrieval
    explanation_service = explanation_service or placeholder_explanation
    safety_service = safety_service or validate_draft
    # `is None`, not `or`: pymongo collections refuse to be used as a true/false value.
    if audit_logs is None:
        audit_logs = get_audit_logs_collection()

    task_id = str(uuid.uuid4())
    report_id = str(uuid.uuid4())

    def audit(agent: str, action: str, status: str, **details) -> None:
        # Ids, outcomes and counts only: never values, test names or draft text.
        log_event(
            audit_logs,
            task_id=task_id,
            report_id=report_id,
            user_id=user_id,
            agent=agent,
            action=action,
            status=status,
            details=details,
        )

    def finish(status: Literal["approved", "fallback", "failed"], **fields) -> CoordinatorResult:
        audit_status = {"approved": "approved", "fallback": "rejected", "failed": "error"}[status]
        audit("coordinator", "analyze_report", audit_status, outcome=status)
        return CoordinatorResult(task_id=task_id, report_id=report_id, user_id=user_id, status=status, **fields)

    # 1. Document stage
    try:
        document = document_service(filename, content)
    except DocumentExtractionError as exc:
        # These messages are fixed, user-facing strings with no patient content.
        audit("document_agent", "extract", "error", error=str(exc))
        return finish("failed", message=str(exc))
    except Exception as exc:
        logger.exception("Document extraction failed for task %s", task_id)
        audit("document_agent", "extract", "error", error_type=type(exc).__name__)
        return finish("failed", message=DOCUMENT_UNAVAILABLE_MESSAGE)

    if not document.results:
        audit("document_agent", "extract", "error", error="no_results")
        return finish("failed", report_type=document.report_type, message=NO_RESULTS_MESSAGE)
    audit(
        "document_agent",
        "extract",
        "success",
        extraction_method=document.extraction_method,
        result_count=len(document.results),
    )

    # 2. Status stage (pure code)
    results = [
        AnalyzedLabResult(
            **r.model_dump(),
            status=compute_status(r.value, r.reference_range),
            warning=NEEDS_VERIFICATION_WARNING if r.needs_verification else None,
        )
        for r in document.results
    ]
    findings = [
        Finding(test=r.test, value=r.value, unit=r.unit, reference_range=r.reference_range, status=r.status)
        for r in results
    ]
    audit(
        "coordinator",
        "compute_status",
        "success",
        computed=sum(r.status is not None for r in results),
        unknown=sum(r.status is None for r in results),
    )

    # 3. Retrieval stage
    requested = {f.test.lower() for f in findings}
    try:
        raw_sources = retrieval_service([f.test for f in findings])
        retrieved_sources = [
            source
            for source in (RetrievedSource.model_validate(s) for s in raw_sources or [])
            # Ignore sources for tests we didn't ask about, and empty ones.
            if source.test_name.lower() in requested and source.information
        ]
        tests_with_sources = {s.test_name.lower() for s in retrieved_sources}
        audit(
            "retrieval_agent",
            "retrieve",
            "success",
            tests_requested=len(requested),
            tests_with_sources=len(tests_with_sources),
        )
    except Exception as exc:
        logger.exception("Retrieval failed for task %s", task_id)
        audit("retrieval_agent", "retrieve", "error", error_type=type(exc).__name__)
        retrieved_sources, tests_with_sources = [], set()

    # Only tests with evidence are sent for explanation, so nothing is explained from thin air.
    findings_to_explain = [f for f in findings if f.test.lower() in tests_with_sources]
    common = {"report_type": document.report_type, "results": results}

    # 4-6. Explanation -> draft -> Safety, with regeneration on rejection.
    instruction = None
    for attempt in range(1, MAX_EXPLANATION_RETRIES + 2):
        explained: list[ExplainedFinding] = []
        if findings_to_explain:
            try:
                output = explanation_service(findings_to_explain, retrieved_sources, instruction)
                explained = ExplanationOutput.model_validate(output).findings
                audit("explanation_agent", "explain", "success", attempt=attempt)
            except Exception as exc:
                logger.exception("Explanation failed for task %s (attempt %s)", task_id, attempt)
                audit("explanation_agent", "explain", "error", attempt=attempt, error_type=type(exc).__name__)
                return finish("fallback", message=EXPLANATION_UNAVAILABLE_MESSAGE, **common)

        draft = build_draft_response(findings, explained, tests_with_sources)

        try:
            decision = safety_service(
                SafetyValidateRequest(
                    task_id=task_id,
                    report_id=report_id,
                    user_id=user_id,
                    # The Document Agent's original numbers, not the explanation's string copies.
                    original_result=[LabResult.model_validate(r.model_dump()) for r in document.results],
                    retrieved_sources=retrieved_sources,
                    draft_response=draft,
                )
            )
        except Exception as exc:
            logger.exception("Safety validation failed for task %s (attempt %s)", task_id, attempt)
            audit("safety_agent", "validate", "error", attempt=attempt, error_type=type(exc).__name__)
            return finish("fallback", message=FALLBACK_MESSAGE, **common)

        if isinstance(decision, ApprovedResponse) and decision.approved:
            audit("safety_agent", "validate", "approved", attempt=attempt)
            return finish("approved", final_response=decision.response, **common)

        reason = getattr(decision, "reason", "unknown")
        audit("safety_agent", "validate", "rejected", attempt=attempt, reason=reason)
        # Regenerating can't help if nothing came from the Explanation service.
        if not findings_to_explain:
            break
        instruction = REGENERATION_INSTRUCTIONS.get(reason, DEFAULT_REGENERATION_INSTRUCTION)

    # Every attempt was rejected: return the fallback, never the unapproved draft.
    return finish("fallback", message=FALLBACK_MESSAGE, **common)
