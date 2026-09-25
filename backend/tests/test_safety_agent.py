import pytest
from fastapi.testclient import TestClient

from agents.safety_agent import (
    DISCLAIMER,
    check_diagnosis,
    check_disclaimer,
    check_medication,
    check_unsupported_claims,
    check_values,
)
from main import app

client = TestClient(app)

ORIGINAL_RESULT = [
    {"test": "Hemoglobin", "value": 11.2, "unit": "g/dL", "reference_range": "12.0-15.5"},
]

RETRIEVED_SOURCES = [
    {
        "test_name": "Hemoglobin",
        "information": {
            "description": "Hemoglobin is a protein in red blood cells that carries oxygen.",
            "low_causes": ["iron deficiency", "blood loss"],
        },
        "sources": ["https://medlineplus.gov/lab-tests/hemoglobin-test/"],
    }
]

CLEAN_BODY = (
    "Your hemoglobin is 11.2 g/dL, which is below the reference range of 12.0-15.5 g/dL. "
    "Hemoglobin is a protein in red blood cells that carries oxygen around your body. "
    "Common reasons for a low level include iron deficiency and blood loss."
)
CLEAN_DRAFT = f"{CLEAN_BODY}\n\n{DISCLAIMER}"


def validate(draft):
    payload = {
        "task_id": "task-1",
        "report_id": "report-1",
        "user_id": "user-1",
        "original_result": ORIGINAL_RESULT,
        "retrieved_sources": RETRIEVED_SOURCES,
        "draft_response": draft,
    }
    response = client.post("/agents/safety/validate", json=payload)
    assert response.status_code == 200
    return response.json()


# --- Endpoint: one test per required scenario ------------------------------------


def test_clean_draft_with_disclaimer_is_approved():
    body = validate(CLEAN_DRAFT)

    assert body == {
        "approved": True,
        "checks": {
            "patient_values_verified": True,
            "diagnosis_detected": False,
            "medication_detected": False,
            "unsupported_claim_detected": False,
            "disclaimer_present": True,
        },
        "response": CLEAN_DRAFT,
    }


@pytest.mark.parametrize(
    "draft, reason",
    [
        (f"Your hemoglobin is 11.2 g/dL, so you have anemia. {DISCLAIMER}", "diagnosis_detected"),
        (f"Your hemoglobin is 11.2 g/dL. You should take iron supplements. {DISCLAIMER}", "medication_detected"),
        (CLEAN_DRAFT.replace("11.2", "17.2"), "patient_values_verified"),
        (CLEAN_BODY, "disclaimer_present"),
        (
            f"{CLEAN_BODY} Low hemoglobin is also commonly caused by kidney disease. {DISCLAIMER}",
            "unsupported_claim_detected",
        ),
    ],
    ids=["diagnosis", "medication", "wrong-value", "missing-disclaimer", "unsupported-claim"],
)
def test_unsafe_draft_is_rejected(draft, reason):
    assert validate(draft) == {"approved": False, "reason": reason, "action": "regenerate"}


# --- check_values -----------------------------------------------------------------


@pytest.mark.parametrize(
    "draft",
    [
        "Your hemoglobin is 11.2 g/dL.",
        "Your hemoglobin is 11.2g/dL.",
        "Your hemoglobin is 11.20.",
        "Hemoglobin (normal 12.0-15.5) came back at 11.2.",
        "Hemoglobin carries oxygen.",  # mentions the test but states no numbers
    ],
)
def test_check_values_accepts_matching_values(draft):
    assert check_values(draft, ORIGINAL_RESULT) is True


def test_check_values_rejects_wrong_value():
    assert check_values("Your hemoglobin is 17.2 g/dL.", ORIGINAL_RESULT) is False


def test_check_values_ignores_digits_inside_units():
    results = [{"test": "Platelets", "value": 250, "unit": "10^9/L", "reference_range": "150-400"}]
    assert check_values("Your platelets are 250 x10^9/L.", results) is True


# --- check_diagnosis --------------------------------------------------------------


@pytest.mark.parametrize(
    "draft",
    ["You have anemia.", "YOU HAVE ANEMIA", "You were diagnosed with diabetes.", "You are anemic."],
)
def test_check_diagnosis_flags_diagnoses(draft):
    assert check_diagnosis(draft) is True


@pytest.mark.parametrize(
    "draft",
    [
        "If you have any questions, ask your doctor.",
        "The HbA1c test is used to monitor diabetes.",
        DISCLAIMER,
    ],
)
def test_check_diagnosis_allows_educational_text(draft):
    assert check_diagnosis(draft) is False


# --- check_medication -------------------------------------------------------------


@pytest.mark.parametrize(
    "draft",
    ["Take iron supplements daily.", "Your doctor may prescribe metformin.", "The usual dosage is low."],
)
def test_check_medication_flags_advice(draft):
    assert check_medication(draft) is True


@pytest.mark.parametrize(
    "draft",
    [
        "Take this report to your doctor.",
        "Your body takes in vitamin B12 from food.",
        "Your cholesterol result is in range.",
        DISCLAIMER,
    ],
)
def test_check_medication_allows_non_advice(draft):
    assert check_medication(draft) is False


# --- check_unsupported_claims -----------------------------------------------------


def test_check_unsupported_claims_passes_when_everything_traces_back():
    assert check_unsupported_claims(CLEAN_DRAFT, ORIGINAL_RESULT, RETRIEVED_SOURCES) is False


@pytest.mark.parametrize(
    "extra",
    ["It can also be caused by kidney disease.", "About 30% of adults have a low level."],
    ids=["unknown-term", "unknown-number"],
)
def test_check_unsupported_claims_flags_untraceable_content(extra):
    draft = f"{CLEAN_BODY} {extra}"
    assert check_unsupported_claims(draft, ORIGINAL_RESULT, RETRIEVED_SOURCES) is True


def test_check_unsupported_claims_uses_whole_words():
    # "liver" must not match inside "delivery".
    draft = "Thanks for the delivery of your sample."
    assert check_unsupported_claims(draft, ORIGINAL_RESULT, RETRIEVED_SOURCES) is False


# --- check_disclaimer -------------------------------------------------------------


def test_check_disclaimer_present_ignoring_case_and_line_breaks():
    reformatted = DISCLAIMER.upper().replace(" ", "\n", 3)
    assert check_disclaimer(f"Some text. {reformatted}") is True


def test_check_disclaimer_missing():
    assert check_disclaimer(CLEAN_BODY) is False
