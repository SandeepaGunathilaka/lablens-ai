"""Coordinator: runs an uploaded lab report through all four agents.

Pipeline: Document -> status (computed here, in code) -> Retrieval -> Explanation
-> Safety, regenerating the explanation up to MAX_EXPLANATION_RETRIES times if Safety
rejects it. Only a Safety-approved draft is ever returned to the caller.

Agents are called as plain Python functions through small, injectable service
interfaces, so tests can pass fakes instead of the real agents.
"""

import logging
import re
import uuid
from collections import Counter
from collections.abc import Callable
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pymongo.collection import Collection

from agents.document_agent import DocumentExtractionError, DocumentExtractionResponse, extract_document
from agents.keyword_retriever import KeywordRetriever
from agents.knowledge_base import KnowledgeBaseValidationError
from agents.lay_terms import match_lay_term
from agents.retrieval_agent import MedicalRetrievalAgent
from agents.retrieval_models import RetrievalRequest, RetrievalResponse
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
from explanation_agent.models import (
    ExplainedFinding,
    ExplanationFinding,
    ExplanationRequest,
    ExplanationResponse,
)
from explanation_agent.service import ExplanationService, build_explanation_service
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
IMPLAUSIBLE_VALUE_WARNING = (
    "This value is outside what is physically possible for this test, so it may have been misread. "
    "Please verify it against the original report."
)

# Physiological limits per (test, unit), far wider than any reference range. A value
# outside them is treated as a likely extraction error, never as a high or low result.
# Units are compared lowercase without spaces; an unlisted unit is not checked, because
# the same test is reported on different scales (g/dL vs g/L).
PLAUSIBLE_LIMITS: dict[tuple[str, str], tuple[float, float]] = {
    ("hemoglobin", "g/dl"): (1.0, 25.0),
    ("hemoglobin", "g/l"): (10.0, 250.0),
    ("wbc", "x10^9/l"): (0.1, 1000.0),
    ("wbc", "10^9/l"): (0.1, 1000.0),
    ("wbc", "x10^3/ul"): (0.1, 1000.0),
    ("platelets", "x10^9/l"): (1.0, 5000.0),
    ("platelets", "10^9/l"): (1.0, 5000.0),
    ("platelets", "x10^3/ul"): (1.0, 5000.0),
    ("total cholesterol", "mg/dl"): (20.0, 2000.0),
    ("total cholesterol", "mmol/l"): (0.5, 50.0),
    ("hdl", "mg/dl"): (1.0, 300.0),
    ("hdl", "mmol/l"): (0.03, 8.0),
    ("hdl cholesterol", "mg/dl"): (1.0, 300.0),
    ("hdl cholesterol", "mmol/l"): (0.03, 8.0),
    ("ldl", "mg/dl"): (1.0, 1500.0),
    ("ldl", "mmol/l"): (0.03, 40.0),
    ("ldl cholesterol", "mg/dl"): (1.0, 1500.0),
    ("ldl cholesterol", "mmol/l"): (0.03, 40.0),
    ("triglycerides", "mg/dl"): (5.0, 20000.0),
    ("triglycerides", "mmol/l"): (0.05, 230.0),
}

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
# Same shape as MedicalRetrievalAgent.retrieve, so an agent's bound method can be passed in.
RetrievalService = Callable[[RetrievalRequest], RetrievalResponse]
# Same shape as ExplanationService.explain, so a service's bound method can be passed in.
ExplanationFn = Callable[[ExplanationRequest], ExplanationResponse]
SafetyService = Callable[[SafetyValidateRequest], ApprovedResponse | RejectedResponse]


@lru_cache(maxsize=1)
def _default_retrieval_agent() -> MedicalRetrievalAgent:
    # Created on first use, not at import: building it loads the knowledge base. If
    # construction fails, nothing is cached and the retrieval stage reports an error.
    return MedicalRetrievalAgent()


def default_retrieval(request: RetrievalRequest) -> RetrievalResponse:
    """The real Retrieval Agent (Member 2)."""
    return _default_retrieval_agent().retrieve(request)


_RECORD_TEST_LINE = re.compile(r"\ATest: (?P<name>[^\n]+)")


@lru_cache(maxsize=1)
def _approved_vocabulary() -> KeywordRetriever:
    return KeywordRetriever()


def _canonical_test_name(name: str) -> str | None:
    """The curated test a name or approved alias refers to, or None if it is not approved vocabulary."""
    try:
        document = _approved_vocabulary().retrieve(match_lay_term(name) or name)
    except (KnowledgeBaseValidationError, OSError):
        return None
    return document.test_name if document is not None else None


def _evidence_matches_request(requested: str, information: str) -> bool:
    """False when a passage's "Test:" line names a different curated test than was requested.

    Only names that resolve through approved vocabulary can be compared; a semantic match
    for free text has no expected record and is accepted as retrieved.
    """
    record = _RECORD_TEST_LINE.match(information)
    if record is None:
        return True
    expected = _canonical_test_name(requested)
    actual = _canonical_test_name(record.group("name"))
    return expected is None or actual is None or expected == actual


def retrieval_response_to_sources(response: RetrievalResponse) -> list[RetrievedSource]:
    """Convert the Retrieval Agent's response into the sources the Coordinator passes on.

    Results with found=False are skipped, so those tests take the existing
    insufficient-information path. So are results whose record is for a different
    test than the one requested. Each test appears at most once.
    """
    sources: dict[str, RetrievedSource] = {}
    for result in response.results:
        if not result.found or result.test_name.lower() in sources:
            continue
        if not all(_evidence_matches_request(result.test_name, m.information) for m in result.matches):
            logger.warning("Discarded retrieved evidence for %r: the record is for a different test", result.test_name)
            continue
        sources[result.test_name.lower()] = RetrievedSource(
            test_name=result.test_name,
            information={"passages": [match.information for match in result.matches]},
            sources=[s.model_dump(mode="json") for match in result.matches for s in match.sources],
        )
    return list(sources.values())


@lru_cache(maxsize=1)
def _default_explanation_service() -> ExplanationService:
    # Built once on first use and reused. Reads EXPLANATION_PROVIDER / GEMINI_API_KEY;
    # without a key it runs in "unavailable" mode and returns safe text, never raising.
    return build_explanation_service()


def default_explanation(request: ExplanationRequest) -> ExplanationResponse:
    """The real Explanation Agent (Member 3)."""
    return _default_explanation_service().explain(request)


def rejection_note(reason: str) -> str:
    """One rejection_feedback entry: which Safety check failed and what to change."""
    instruction = REGENERATION_INSTRUCTIONS.get(reason, DEFAULT_REGENERATION_INSTRUCTION)
    return f"Safety check '{reason}' failed. {instruction}"


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


def is_plausible_value(test: str, value: float, unit: str | None) -> bool:
    """False for a negative value, or one outside PLAUSIBLE_LIMITS for its test and unit."""
    if value < 0:
        return False
    key = (test.strip().lower(), re.sub(r"\s+", "", unit or "").lower())
    limits = PLAUSIBLE_LIMITS.get(key)
    return limits is None or limits[0] <= value <= limits[1]


def analyze_result(result) -> "AnalyzedLabResult":
    """Status and warning for one extracted value. Implausible values get no status."""
    plausible = is_plausible_value(result.test, result.value, result.unit)
    if not plausible:
        warning = IMPLAUSIBLE_VALUE_WARNING
    elif result.needs_verification:
        warning = NEEDS_VERIFICATION_WARNING
    else:
        warning = None
    return AnalyzedLabResult(
        **{**result.model_dump(), "needs_verification": result.needs_verification or not plausible},
        status=compute_status(result.value, result.reference_range) if plausible else None,
        warning=warning,
    )


# --- Draft building --------------------------------------------------------------


def _format_number(value: float) -> str:
    """11.2 -> "11.2", 250.0 -> "250" (never scientific notation)."""
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return text or "0"


def _value_text(value: float, unit: str | None) -> str:
    number = _format_number(value)
    return f"{number} {unit}" if unit else number


def _result_line(result: AnalyzedLabResult, result_text: str) -> str:
    """The value/range/status sentence. Status is always the one computed in code."""
    line = f"{result.test}: your result is {result_text}"
    if result.reference_range:
        line += f"; the range on your report is {result.reference_range}"
    if result.status:
        line += f", which is {result.status}"
    return line + "."


def _match_explained(
    index: int, result: AnalyzedLabResult, explained: list[ExplainedFinding]
) -> ExplainedFinding | None:
    """The explanation for this result: by position (the agent keeps request order), else by name."""
    if index < len(explained) and explained[index].test_name.lower() == result.test.lower():
        return explained[index]
    return next((e for e in explained if e.test_name.lower() == result.test.lower()), None)


def build_draft_response(
    results: list[AnalyzedLabResult],
    explained: list[ExplainedFinding],
    tests_with_sources: set[str],
) -> str:
    """Join every finding into one draft and append the standard DISCLAIMER.

    The result text comes from the explanation's ``result`` field and the status from
    code. Tests without retrieved sources get INSUFFICIENT_INFORMATION_MESSAGE, and the
    agent's prose for them is only used if the agent itself marked it insufficient (its
    fixed, claim-free wording): generated text without evidence is never included.
    """
    sections = []
    for index, result in enumerate(results):
        item = _match_explained(index, result, explained)
        result_text = item.result if item else _value_text(result.value, result.unit)
        parts = [_result_line(result, result_text)]
        has_sources = result.test.lower() in tests_with_sources
        if not has_sources:
            parts.append(INSUFFICIENT_INFORMATION_MESSAGE)
        if item and (has_sources or item.insufficient_information):
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
    explanation_service: ExplanationFn | None = None,
    safety_service: SafetyService | None = None,
    audit_logs: Collection | None = None,
) -> CoordinatorResult:
    """Run an uploaded report through the full pipeline. Never raises for agent failures."""
    document_service = document_service or extract_document
    retrieval_service = retrieval_service or default_retrieval
    explanation_service = explanation_service or default_explanation
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
    results = [analyze_result(r) for r in document.results]
    audit(
        "coordinator",
        "compute_status",
        "success",
        computed=sum(r.status is not None for r in results),
        unknown=sum(r.status is None for r in results),
        implausible=sum(r.warning == IMPLAUSIBLE_VALUE_WARNING for r in results),
    )

    # 3. Retrieval stage
    requested = {r.test.lower() for r in results}
    try:
        request = RetrievalRequest(
            task_id=task_id, report_id=report_id, user_id=user_id, test_names=[r.test for r in results]
        )
        response = RetrievalResponse.model_validate(retrieval_service(request))
        # Evidence tagged for another task/report/user must never reach this report.
        if (response.task_id, response.report_id, response.user_id) != (task_id, report_id, user_id):
            raise ValueError("Retrieval response IDs do not match the request")
        retrieved_sources = [
            source
            for source in retrieval_response_to_sources(response)
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

    common = {"report_type": document.report_type, "results": results}

    # 4-6. Explanation -> draft -> Safety, with regeneration on rejection.
    # Every finding is sent each time, with all sources; the Explanation Agent matches
    # sources to findings itself and answers "insufficient" for tests without any.
    rejection_feedback: list[str] = []
    for attempt in range(1, MAX_EXPLANATION_RETRIES + 2):
        try:
            explanation_request = ExplanationRequest(
                task_id=task_id,
                report_id=report_id,
                user_id=user_id,
                findings=[
                    ExplanationFinding(
                        test=r.test, value=r.value, unit=r.unit, reference_range=r.reference_range, status=r.status
                    )
                    for r in results
                ],
                retrieved_sources=retrieved_sources,
                rejection_feedback=list(rejection_feedback),
            )
            explanation = ExplanationResponse.model_validate(explanation_service(explanation_request))
            if (explanation.task_id, explanation.report_id, explanation.user_id) != (task_id, report_id, user_id):
                raise ValueError("Explanation response IDs do not match the request")
            explained = explanation.findings
            audit(
                "explanation_agent",
                "explain",
                "success",
                attempt=attempt,
                generation_modes=dict(sorted(Counter(f.generation_mode for f in explained).items())),
            )
        except Exception as exc:
            # A real failure, not the agent's normal "unavailable" mode (which returns safe text).
            logger.exception("Explanation failed for task %s (attempt %s)", task_id, attempt)
            audit("explanation_agent", "explain", "error", attempt=attempt, error_type=type(exc).__name__)
            return finish("fallback", message=EXPLANATION_UNAVAILABLE_MESSAGE, **common)

        draft = build_draft_response(results, explained, tests_with_sources)

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
        # Each rejection is added to the feedback the next explanation attempt receives.
        rejection_feedback.append(rejection_note(reason))

    # Every attempt was rejected: return the fallback, never the unapproved draft.
    return finish("fallback", message=FALLBACK_MESSAGE, **common)
