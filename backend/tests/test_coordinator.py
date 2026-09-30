import pytest

from agents.document_agent import DocumentExtractionError, DocumentExtractionResponse, ExtractedLabResult
from agents.hybrid_retriever import HybridRetrievalResult
from agents.knowledge_base import KnowledgeDocument, KnowledgeSource
from agents.retrieval_agent import MedicalRetrievalAgent
from agents.retrieval_models import RetrievalResponse
from agents.safety_agent import DISCLAIMER, check_disclaimer, validate_draft
from coordinator import (
    EXPLANATION_UNAVAILABLE_MESSAGE,
    FALLBACK_MESSAGE,
    INSUFFICIENT_INFORMATION_MESSAGE,
    NEEDS_VERIFICATION_WARNING,
    REGENERATION_INSTRUCTIONS,
    ExplainedFinding,
    ExplanationOutput,
    Finding,
    analyze_report,
    build_draft_response,
    compute_status,
    retrieval_response_to_sources,
)

HEMOGLOBIN = ExtractedLabResult(
    test="Hemoglobin", value=11.2, unit="g/dL", reference_range="12.0-15.5",
    confidence=0.95, needs_verification=False,
)
PLATELETS_UNVERIFIED = ExtractedLabResult(
    test="Platelets", value=250, unit=None, reference_range="150-400",
    confidence=0.5, needs_verification=True,
)

SAFE_TEXT = "Hemoglobin is a protein in red blood cells that carries oxygen."
UNSAFE_TEXT = "This means you have anemia."

# A hand-built knowledge-base entry, so tests need no real KB files or vector index.
HEMOGLOBIN_DOCUMENT = KnowledgeDocument(
    id="hemoglobin",
    test_name="Hemoglobin",
    aliases=["Hgb"],
    report_type="cbc",
    title="Hemoglobin test",
    definition="A blood test.",
    what_it_measures=SAFE_TEXT,
    general_information="Results are compared with a reference range.",
    source=KnowledgeSource(
        publisher="MedlinePlus",
        title="Hemoglobin Test",
        url="https://medlineplus.gov/lab-tests/hemoglobin-test/",
        accessed_date="2026-09-01",
    ),
)


# --- Fakes -------------------------------------------------------------------------


def fake_document(*results, report_type="cbc"):
    def service(filename, content):
        return DocumentExtractionResponse(
            extraction_method="pdf_text", report_type=report_type, results=list(results)
        )
    return service


class FakeHybridRetriever:
    """Stands in for HybridRetriever (no ChromaDB, no embeddings): finds only the given documents."""

    def __init__(self, *documents, error=None):
        self.documents = {d.test_name.lower(): d for d in documents}
        self.error = error
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        if self.error:
            raise self.error
        document = self.documents.get(query.lower())
        if document is None:
            return HybridRetrievalResult(None, False, "none", None)
        return HybridRetrievalResult(document, True, "keyword", None)


def real_retrieval(*documents, error=None):
    """The real MedicalRetrievalAgent's retrieve method, backed by a fake hybrid retriever."""
    return MedicalRetrievalAgent(FakeHybridRetriever(*documents, error=error)).retrieve


class FakeExplanation:
    """Returns the given texts in order (the last one repeats) and records each call."""

    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls = []

    def __call__(self, findings, retrieved_sources, instruction):
        self.calls.append({"findings": findings, "sources": retrieved_sources, "instruction": instruction})
        text = self.texts[min(len(self.calls), len(self.texts)) - 1]
        return ExplanationOutput(
            findings=[
                ExplainedFinding(test_name=f.test, result=str(f.value), status=f.status, explanation=text)
                for f in findings
            ]
        )


def raising(*args, **kwargs):
    raise RuntimeError("service is down")


def run(audit_logs, **services):
    services.setdefault("document_service", fake_document(HEMOGLOBIN))
    services.setdefault("retrieval_service", real_retrieval(HEMOGLOBIN_DOCUMENT))
    return analyze_report("report.pdf", b"%PDF-fake", "user-1", audit_logs=audit_logs, **services)


# --- compute_status ----------------------------------------------------------------


@pytest.mark.parametrize(
    "value, reference_range, expected",
    [
        (11.9, "12-16", "low"),
        (12.0, "12-16", "normal"),  # lower bound is inclusive
        (14.0, "12.0 - 16.0", "normal"),
        (16.0, "12 to 16", "normal"),  # upper bound is inclusive
        (16.1, "12 to 16", "high"),
        (199, "<200", "normal"),
        (200, "<200", "high"),  # "<" is strict
        (201, "<200", "high"),
        (200, "<=200", "normal"),  # "<=" is inclusive
        (39.9, ">40", "low"),
        (40, ">40", "low"),  # ">" is strict
        (40, ">=40", "normal"),  # ">=" is inclusive
        (55, "> 40", "normal"),
        (10, "12–16", "low"),  # en dash, as some PDFs render it
    ],
)
def test_compute_status(value, reference_range, expected):
    assert compute_status(value, reference_range) == expected


def test_compute_status_value_equal_to_strict_upper_bound_is_high():
    assert compute_status(200, "<200") == "high"


def test_compute_status_value_equal_to_strict_lower_bound_is_low():
    assert compute_status(40, ">40") == "low"


@pytest.mark.parametrize(
    "reference_range",
    [None, "", "   ", "normal", "12-", "see note", "12-16 g/dL", "16-12", "-2 to 2", "<"],
)
def test_compute_status_is_none_when_range_missing_or_unparseable(reference_range):
    assert compute_status(13, reference_range) is None


# --- Draft --------------------------------------------------------------------------


def test_draft_ends_with_disclaimer_and_uses_code_computed_status():
    finding = Finding(test="Hemoglobin", value=11.2, unit="g/dL", reference_range="12.0-15.5", status="low")
    # The explanation claims "normal"; the draft must use the status computed in code.
    explained = [ExplainedFinding(test_name="Hemoglobin", result="11.2", status="normal", explanation=SAFE_TEXT)]

    draft = build_draft_response([finding], explained, tests_with_sources={"hemoglobin"})

    assert draft.endswith(DISCLAIMER)
    assert check_disclaimer(draft)
    assert "which is low" in draft
    assert "normal" not in draft
    assert SAFE_TEXT in draft


# --- Pipeline -----------------------------------------------------------------------


def test_happy_path_is_approved(audit_logs):
    safety_requests = []

    def spying_safety(request):
        safety_requests.append(request)
        return validate_draft(request)

    result = run(audit_logs, explanation_service=FakeExplanation(SAFE_TEXT), safety_service=spying_safety)

    assert result.status == "approved"
    assert result.report_type == "cbc"
    assert result.message is None
    assert SAFE_TEXT in result.final_response
    assert result.final_response.endswith(DISCLAIMER)
    assert result.results[0].status == "low"
    # Safety received the Document Agent's original numbers.
    assert [(r.test, r.value) for r in safety_requests[0].original_result] == [("Hemoglobin", 11.2)]


def test_placeholder_explanation_passes_real_safety(audit_logs):
    result = run(audit_logs)  # placeholder explanation, real safety

    assert result.status == "approved"
    assert "Your Hemoglobin value is 11.2 g/dL; the range on your report is 12.0-15.5." in result.final_response


def test_rejected_once_then_regenerated_draft_is_approved(audit_logs):
    explanation = FakeExplanation(UNSAFE_TEXT, SAFE_TEXT)

    result = run(audit_logs, explanation_service=explanation)

    assert result.status == "approved"
    assert UNSAFE_TEXT not in result.final_response
    assert [c["instruction"] for c in explanation.calls] == [None, REGENERATION_INSTRUCTIONS["diagnosis_detected"]]


def test_rejected_every_attempt_returns_fallback_without_draft(audit_logs):
    explanation = FakeExplanation(UNSAFE_TEXT)

    result = run(audit_logs, explanation_service=explanation)

    assert len(explanation.calls) == 3  # 1 attempt + 2 retries
    assert result.status == "fallback"
    assert result.message == FALLBACK_MESSAGE
    assert result.final_response is None
    assert "anemia" not in result.model_dump_json()  # the unapproved draft is nowhere in the result
    assert result.results[0].value == 11.2  # extracted values are still returned


def test_safety_service_crash_returns_fallback(audit_logs):
    result = run(audit_logs, explanation_service=FakeExplanation(SAFE_TEXT), safety_service=raising)

    assert result.status == "fallback"
    assert result.message == FALLBACK_MESSAGE
    assert result.final_response is None


@pytest.mark.parametrize(
    "retrieval_service",
    [real_retrieval(), real_retrieval(error=RuntimeError("index unavailable")), raising],
    ids=["found-false", "retriever-raises", "service-raises"],
)
def test_missing_sources_give_insufficient_information_message(audit_logs, retrieval_service):
    explanation = FakeExplanation(SAFE_TEXT)

    result = run(audit_logs, retrieval_service=retrieval_service, explanation_service=explanation)

    assert result.status == "approved"
    assert INSUFFICIENT_INFORMATION_MESSAGE in result.final_response
    # With no evidence, nothing is sent to be explained, so nothing can be invented.
    assert explanation.calls == []
    assert SAFE_TEXT not in result.final_response


def test_only_tests_with_sources_are_explained(audit_logs):
    explanation = FakeExplanation(SAFE_TEXT)

    result = run(
        audit_logs,
        document_service=fake_document(HEMOGLOBIN, PLATELETS_UNVERIFIED),
        explanation_service=explanation,
    )

    assert [f.test for f in explanation.calls[0]["findings"]] == ["Hemoglobin"]
    platelets_section = result.final_response.split("\n\n")[1]
    assert platelets_section.startswith("Platelets:")
    assert INSUFFICIENT_INFORMATION_MESSAGE in platelets_section


def test_explanation_service_raising_still_returns_extracted_data(audit_logs):
    result = run(audit_logs, explanation_service=raising)

    assert result.status == "fallback"
    assert result.message == EXPLANATION_UNAVAILABLE_MESSAGE
    assert result.final_response is None
    assert [(r.test, r.value, r.status) for r in result.results] == [("Hemoglobin", 11.2, "low")]


def test_needs_verification_warning_passes_through(audit_logs):
    result = run(audit_logs, document_service=fake_document(HEMOGLOBIN, PLATELETS_UNVERIFIED))

    warnings = {r.test: r.warning for r in result.results}
    assert warnings == {"Hemoglobin": None, "Platelets": NEEDS_VERIFICATION_WARNING}


def test_document_extraction_error_returns_failed_result(audit_logs):
    def bad_document(filename, content):
        raise DocumentExtractionError("The uploaded file is empty.")

    result = run(audit_logs, document_service=bad_document)

    assert result.status == "failed"
    assert result.message == "The uploaded file is empty."
    assert result.results == []
    assert result.final_response is None


def test_unexpected_document_error_returns_failed_result(audit_logs):
    result = run(audit_logs, document_service=raising)

    assert result.status == "failed"
    assert result.message and "service is down" not in result.message


# --- Audit ----------------------------------------------------------------------------


def test_audit_entries_per_stage_with_consistent_ids_and_no_patient_content(audit_logs):
    result = run(audit_logs, explanation_service=FakeExplanation(UNSAFE_TEXT, SAFE_TEXT))

    entries = list(audit_logs.find({}, {"_id": 0}).sort([("timestamp", 1), ("_id", 1)]))
    assert [(e["agent"], e["action"], e["status"]) for e in entries] == [
        ("document_agent", "extract", "success"),
        ("coordinator", "compute_status", "success"),
        ("retrieval_agent", "retrieve", "success"),
        ("explanation_agent", "explain", "success"),
        ("safety_agent", "validate", "rejected"),
        ("explanation_agent", "explain", "success"),
        ("safety_agent", "validate", "approved"),
        ("coordinator", "analyze_report", "approved"),
    ]
    assert {(e["task_id"], e["report_id"], e["user_id"]) for e in entries} == {
        (result.task_id, result.report_id, "user-1")
    }
    for entry in entries:
        details = str(entry["details"]).lower()
        for patient_content in ["hemoglobin", "11.2", "12.0-15.5", "g/dl", "anemia", "protein"]:
            assert patient_content not in details


def test_failed_document_stage_is_audited(audit_logs):
    result = run(audit_logs, document_service=raising)

    entries = list(audit_logs.find({"task_id": result.task_id}))
    assert [(e["agent"], e["status"]) for e in entries] == [
        ("document_agent", "error"),
        ("coordinator", "error"),
    ]


# --- Retrieval Agent integration ---------------------------------------------------------


def test_found_result_becomes_sources_for_explanation(audit_logs):
    hybrid = FakeHybridRetriever(HEMOGLOBIN_DOCUMENT)
    agent = MedicalRetrievalAgent(hybrid)
    requests = []

    def spying_retrieval(request):
        requests.append(request)
        return agent.retrieve(request)

    explanation = FakeExplanation(SAFE_TEXT)
    result = run(audit_logs, retrieval_service=spying_retrieval, explanation_service=explanation)

    # The RetrievalRequest carries the Coordinator's own ids and the test names.
    request = requests[0]
    assert (request.task_id, request.report_id, request.user_id) == (result.task_id, result.report_id, "user-1")
    assert request.test_names == ["Hemoglobin"]
    assert hybrid.queries == ["Hemoglobin"]

    # The found result reached the Explanation service as a source.
    [source] = explanation.calls[0]["sources"]
    assert source.test_name == "Hemoglobin"
    assert SAFE_TEXT in source.information["passages"][0]
    assert source.sources == [
        {"title": "Hemoglobin Test", "url": "https://medlineplus.gov/lab-tests/hemoglobin-test/"}
    ]
    assert result.status == "approved"
    assert SAFE_TEXT in result.final_response
    assert INSUFFICIENT_INFORMATION_MESSAGE not in result.final_response


def test_found_and_not_found_results_in_one_report(audit_logs):
    explanation = FakeExplanation(SAFE_TEXT)

    result = run(
        audit_logs,
        document_service=fake_document(HEMOGLOBIN, PLATELETS_UNVERIFIED),
        retrieval_service=real_retrieval(HEMOGLOBIN_DOCUMENT),  # knows Hemoglobin only
        explanation_service=explanation,
    )

    assert [s.test_name for s in explanation.calls[0]["sources"]] == ["Hemoglobin"]
    hemoglobin_section, platelets_section = result.final_response.split("\n\n")[:2]
    assert SAFE_TEXT in hemoglobin_section
    assert INSUFFICIENT_INFORMATION_MESSAGE in platelets_section


def test_retrieval_response_to_sources_skips_not_found_and_duplicates():
    response = RetrievalResponse.model_validate({
        "task_id": "t", "report_id": "r", "user_id": "u",
        "results": [
            {"test_name": "Hemoglobin", "found": True, "matches": [
                {"information": "Passage one.", "sources": [{"title": "A", "url": "https://a.example/"}]},
            ]},
            {"test_name": "Platelets", "found": False},
            {"test_name": "hemoglobin", "found": True, "matches": [
                {"information": "Duplicate.", "sources": [{"title": "B", "url": "https://b.example/"}]},
            ]},
        ],
    })

    sources = retrieval_response_to_sources(response)

    assert [(s.test_name, s.information) for s in sources] == [("Hemoglobin", {"passages": ["Passage one."]})]
    assert sources[0].sources == [{"title": "A", "url": "https://a.example/"}]


def test_retrieval_response_for_another_user_is_rejected(audit_logs):
    agent = MedicalRetrievalAgent(FakeHybridRetriever(HEMOGLOBIN_DOCUMENT))

    def wrong_user_retrieval(request):
        return agent.retrieve(request.model_copy(update={"user_id": "someone-else"}))

    explanation = FakeExplanation(SAFE_TEXT)
    result = run(audit_logs, retrieval_service=wrong_user_retrieval, explanation_service=explanation)

    # Treated as a retrieval failure: no evidence is used.
    assert explanation.calls == []
    assert INSUFFICIENT_INFORMATION_MESSAGE in result.final_response


def test_retrieval_exception_is_audited_without_crashing(audit_logs):
    result = run(audit_logs, retrieval_service=real_retrieval(error=RuntimeError("index unavailable")))

    assert result.status == "approved"
    [entry] = list(audit_logs.find({"task_id": result.task_id, "agent": "retrieval_agent"}))
    assert (entry["action"], entry["status"]) == ("retrieve", "error")
    assert entry["details"] == {"error_type": "RuntimeError"}
