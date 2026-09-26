import math
import re
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from pymongo.collection import Collection

from agents import safety_config as config
from database import get_audit_logs_collection
from logging_service import log_event

# Every approved response must contain this exact text (case and line breaks don't matter).
DISCLAIMER = (
    "This summary is for general education only and is not medical advice. "
    "Please talk to your doctor about your results."
)

router = APIRouter(prefix="/agents/safety", tags=["safety-agent"])


# --- Request / response models --------------------------------------------------


class LabResult(BaseModel):
    test: str
    value: float
    unit: str
    reference_range: str


class RetrievedSource(BaseModel):
    test_name: str
    information: dict
    sources: list


class SafetyValidateRequest(BaseModel):
    task_id: str
    report_id: str
    user_id: str
    original_result: list[LabResult]
    retrieved_sources: list[RetrievedSource]
    draft_response: str


class SafetyChecks(BaseModel):
    patient_values_verified: bool
    diagnosis_detected: bool
    medication_detected: bool
    unsupported_claim_detected: bool
    disclaimer_present: bool


class ApprovedResponse(BaseModel):
    approved: Literal[True] = True
    checks: SafetyChecks
    response: str


class RejectedResponse(BaseModel):
    approved: Literal[False] = False
    reason: str
    action: Literal["regenerate"] = "regenerate"


# The value each check must have for the draft to pass, in the order they're reported.
# Order matters: a wrong patient value also looks like an unsupported number, so
# checking values first gives the more specific reason.
PASSING_CHECKS = {
    "patient_values_verified": True,
    "diagnosis_detected": False,
    "medication_detected": False,
    "unsupported_claim_detected": False,
    "disclaimer_present": True,
}


# --- Text helpers ----------------------------------------------------------------


def _term_regex(term: str, plurals: bool = True) -> str:
    """Whole-word regex for one term, tolerating any whitespace between its words."""
    words = r"\s+".join(re.escape(word) for word in term.split())
    suffix = r"(?:s|es)?" if plurals else ""
    # (?<!\w) / (?!\w) act like \b but also work when a term starts or ends with
    # punctuation, e.g. "Hemoglobin (Hb)".
    return rf"(?<!\w){words}{suffix}(?!\w)"


def _compile_terms(terms: list[str], plurals: bool = True) -> re.Pattern:
    """One case-insensitive regex that matches any of the terms."""
    return re.compile("|".join(_term_regex(t, plurals) for t in terms), re.IGNORECASE)


DIAGNOSIS_PHRASES_RE = _compile_terms(config.DIAGNOSIS_PHRASES)
DIAGNOSIS_ASSERTIONS_RE = _compile_terms(config.DIAGNOSIS_ASSERTIONS, plurals=False)
DISEASE_TERMS_RE = _compile_terms(config.DISEASE_TERMS)
MEDICATION_PHRASES_RE = _compile_terms(config.MEDICATION_PHRASES + config.DRUG_NAMES)
MEDICATION_ACTIONS_RE = _compile_terms(config.MEDICATION_ACTIONS, plurals=False)
MEDICATION_TERMS_RE = _compile_terms(config.MEDICATION_TERMS)
# One pattern per term, so we can check each term against the sources individually.
UNSUPPORTED_TERM_RES = [re.compile(_term_regex(t), re.IGNORECASE) for t in config.UNSUPPORTED_CLAIM_TERMS]
DISCLAIMER_RE = re.compile(_term_regex(DISCLAIMER, plurals=False), re.IGNORECASE)

# A number such as 11.2, 150,000 or 7, but not the digits inside words like "B12" or "HbA1c".
NUMBER_RE = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
# Scientific unit multipliers like "x10^9" whose digits aren't measurements.
SCIENTIFIC_UNIT_RE = re.compile(r"[x×*]?\s*10\s*\^\s*-?\d+", re.IGNORECASE)
# List markers at the start of a line, like "1." or "2)".
LIST_MARKER_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _strip_disclaimer(text: str) -> str:
    """Remove the disclaimer so its own wording can't trigger the content checks."""
    return DISCLAIMER_RE.sub(" ", text)


def _sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _numbers(text: str) -> list[float]:
    return [float(match.replace(",", "")) for match in NUMBER_RE.findall(text)]


def _is_known_number(number: float, allowed: list[float]) -> bool:
    return any(math.isclose(number, a, rel_tol=1e-9, abs_tol=1e-9) for a in allowed)


def _draft_body(draft_response: str, original_result: list[dict]) -> str:
    """The draft without the disclaimer, list markers, or numbers that are part of units."""
    text = _strip_disclaimer(draft_response)
    for result in original_result:
        unit = result["unit"]
        if any(ch.isdigit() for ch in unit):
            text = re.sub(re.escape(unit), " ", text, flags=re.IGNORECASE)
    text = SCIENTIFIC_UNIT_RE.sub(" ", text)
    return LIST_MARKER_RE.sub("", text)


def _flatten_text(value) -> list[str]:
    """Every string, key and number inside nested dicts/lists, as strings."""
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in [str(k), *_flatten_text(v)]]
    if isinstance(value, (list, tuple)):
        return [s for item in value for s in _flatten_text(item)]
    if value is None:
        return []
    return [str(value)]


# --- The five checks -------------------------------------------------------------


def check_values(draft_response: str, original_result: list[dict]) -> bool:
    """True if every number stated next to a test name matches the patient's data.

    Works sentence by sentence: in any sentence that mentions a test, each number must
    be that test's value or one of its reference range bounds. Units are ignored,
    so "11.2 g/dL", "11.2g/dL" and "11.20" all match a value of 11.2.
    """
    text = _draft_body(draft_response, original_result)
    for sentence in _sentences(text):
        mentioned = [
            r for r in original_result
            if re.search(_term_regex(r["test"], plurals=False), sentence, re.IGNORECASE)
        ]
        if not mentioned:
            continue
        allowed = [n for r in mentioned for n in [r["value"], *_numbers(r["reference_range"])]]
        if not all(_is_known_number(n, allowed) for n in _numbers(sentence)):
            return False
    return True


def check_diagnosis(draft_response: str) -> bool:
    """True if the draft tells the patient they have a condition.

    Flags unambiguous phrases ("diagnosed with") anywhere, and personal statements
    ("you have") only when a disease name appears in the same sentence.
    """
    text = _strip_disclaimer(draft_response)
    if DIAGNOSIS_PHRASES_RE.search(text):
        return True
    return any(
        DIAGNOSIS_ASSERTIONS_RE.search(s) and DISEASE_TERMS_RE.search(s)
        for s in _sentences(text)
    )


def check_medication(draft_response: str) -> bool:
    """True if the draft gives medication advice.

    Flags dosing words and drug names anywhere, and action verbs ("take") only when a
    medication word ("supplement", "pill", ...) appears in the same sentence.
    """
    text = _strip_disclaimer(draft_response)
    if MEDICATION_PHRASES_RE.search(text):
        return True
    return any(
        MEDICATION_ACTIONS_RE.search(s) and MEDICATION_TERMS_RE.search(s)
        for s in _sentences(text)
    )


def check_unsupported_claims(
    draft_response: str, original_result: list[dict], retrieved_sources: list[dict]
) -> bool:
    """True if the draft mentions a number or medical term that isn't in the evidence.

    The evidence is the patient's results plus the retrieved sources. Every number in
    the draft, and every term from the configured medical vocabulary, must appear there.
    """
    evidence = " ".join(_flatten_text(original_result) + _flatten_text(retrieved_sources))
    allowed_numbers = [r["value"] for r in original_result] + _numbers(evidence)

    text = _draft_body(draft_response, original_result)
    if not all(_is_known_number(n, allowed_numbers) for n in _numbers(text)):
        return True
    return any(term.search(text) and not term.search(evidence) for term in UNSUPPORTED_TERM_RES)


def check_disclaimer(draft_response: str) -> bool:
    """True if the draft contains the required DISCLAIMER."""
    return DISCLAIMER_RE.search(draft_response) is not None


# --- Endpoint --------------------------------------------------------------------


@router.post("/validate", response_model=ApprovedResponse | RejectedResponse)
def validate(payload: SafetyValidateRequest, audit_logs: Collection = Depends(get_audit_logs_collection)):
    draft = payload.draft_response
    results = [r.model_dump() for r in payload.original_result]
    sources = [s.model_dump() for s in payload.retrieved_sources]

    checks = {
        "patient_values_verified": check_values(draft, results),
        "diagnosis_detected": check_diagnosis(draft),
        "medication_detected": check_medication(draft),
        "unsupported_claim_detected": check_unsupported_claims(draft, results, sources),
        "disclaimer_present": check_disclaimer(draft),
    }

    # The first check (in PASSING_CHECKS order) that failed, or None if all passed.
    failed_check = next(
        (name for name, passing_value in PASSING_CHECKS.items() if checks[name] != passing_value),
        None,
    )

    # Reference example of audit logging: one call per decision, ids and outcome only,
    # never the patient's medical content (lab values, draft text).
    details = {"checks": checks}
    if failed_check:
        details["reason"] = failed_check
    log_event(
        audit_logs,
        task_id=payload.task_id,
        report_id=payload.report_id,
        user_id=payload.user_id,
        agent="safety_agent",
        action="validate",
        status="rejected" if failed_check else "approved",
        details=details,
    )

    if failed_check:
        return RejectedResponse(reason=failed_check)
    return ApprovedResponse(checks=SafetyChecks(**checks), response=draft)
