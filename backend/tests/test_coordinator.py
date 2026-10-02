import pytest

import coordinator
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
    AnalyzedLabResult,
    analyze_report,
    build_draft_response,
    compute_status,
    rejection_note,
    retrieval_response_to_sources,
)
from explanation_agent.models import ExplainedFinding, ExplanationResponse
from explanation_agent.service import ExplanationService

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
    """Acts like a model that writes the given texts in order (the last one repeats).

    It answers every finding with generated ("llm") text, even ones with no sources,
    so tests can check the Coordinator never uses ungrounded text. Records each request.
    """

    def __init__(self, *texts):
        self.texts = list(texts)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        text = self.texts[min(len(self.requests), len(self.texts)) - 1]
        return ExplanationResponse(
            task_id=request.task_id,
            report_id=request.report_id,
            user_id=request.user_id,
            findings=[
                ExplainedFinding(
                    test_name=f.test,
                    result=f"{f.value:g} {f.unit or ''}".strip(),
                    status=f.status,
                    what_it_measures="",
                    explanation=text,
                    possible_meaning="",
                    recommended_discussion="",
                    insufficient_information=False,
                    sources_used=[],
                    generation_mode="llm",
                )
                for f in request.findings
            ],
        )


def template_explanation():
    """Member 3's real ExplanationService in deterministic template mode: no model, no key."""
    return ExplanationService(mode="template").explain


def raising(*args, **kwargs):
    raise RuntimeError("service is down")


@pytest.fixture(autouse=True)
def never_call_real_explanation_model(monkeypatch):
    """Fail loudly if a test reaches the real (Gemini-backed) default Explanation service."""
    def forbidden():
        raise AssertionError("tests must inject an explanation_service")

    monkeypatch.setattr(coordinator, "_default_explanation_service", forbidden)


def run(audit_logs, **services):
    services.setdefault("document_service", fake_document(HEMOGLOBIN))
    services.setdefault("retrieval_service", real_retrieval(HEMOGLOBIN_DOCUMENT))
    services.setdefault("explanation_service", template_explanation())
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


def analyzed(test="Hemoglobin", value=11.2, unit="g/dL", reference_range="12.0-15.5", status="low"):
    return AnalyzedLabResult(
        test=test, value=value, unit=unit, reference_range=reference_range,
        confidence=0.95, needs_verification=False, status=status,
    )


def explained_finding(test_name="Hemoglobin", **overrides):
    fields = {
        "test_name": test_name,
        "result": "11.2 g/dL",
        "status": "low",
        "what_it_measures": "",
        "explanation": SAFE_TEXT,
        "possible_meaning": "",
        "recommended_discussion": "",
        "insufficient_information": False,
        "sources_used": ["Hemoglobin Test"],
        "generation_mode": "llm",
    }
    return ExplainedFinding(**{**fields, **overrides})


def test_draft_ends_with_disclaimer_and_uses_code_computed_status():
    # The explanation claims "normal"; the draft must use the status computed in code.
    explained = [explained_finding(status="normal")]

    draft = build_draft_response([analyzed()], explained, tests_with_sources={"hemoglobin"})

    assert draft.endswith(DISCLAIMER)
    assert check_disclaimer(draft)
    assert "which is low" in draft
    assert "normal" not in draft
    assert SAFE_TEXT in draft


def test_draft_uses_the_explanations_result_text():
    explained = [explained_finding(result="11.20 g/dL")]

    draft = build_draft_response([analyzed()], explained, tests_with_sources={"hemoglobin"})

    assert draft.startswith("Hemoglobin: your result is 11.20 g/dL;")


def test_draft_drops_generated_text_for_a_test_without_sources():
    generated = explained_finding(explanation="Invented claim with no evidence.")
    insufficient = explained_finding(
        explanation="There is not enough reliable information.", insufficient_information=True,
        generation_mode="insufficient",
    )

    dropped = build_draft_response([analyzed()], [generated], tests_with_sources=set())
    kept = build_draft_response([analyzed()], [insufficient], tests_with_sources=set())

    assert "Invented claim" not in dropped
    assert INSUFFICIENT_INFORMATION_MESSAGE in dropped
    assert "There is not enough reliable information." in kept
    assert INSUFFICIENT_INFORMATION_MESSAGE in kept


def test_draft_pairs_repeated_tests_by_position():
    results = [analyzed(value=11.2), analyzed(value=12.4)]
    explained = [explained_finding(result="11.2 g/dL"), explained_finding(result="12.4 g/dL")]

    first, second = build_draft_response(results, explained, {"hemoglobin"}).split("\n\n")[:2]

    assert "11.2 g/dL" in first
    assert "12.4 g/dL" in second


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


def test_real_template_explanation_with_sources_flows_into_the_draft(audit_logs):
    result = run(audit_logs)  # Member 3's real ExplanationService (template mode), real Safety

    assert result.status == "approved"
    hemoglobin_section = result.final_response.split("\n\n")[0]
    # Template text quoting the retrieved source, plus the agent's fixed sentences.
    assert 'From "Hemoglobin Test":' in hemoglobin_section
    assert SAFE_TEXT in hemoglobin_section
    assert "The recorded result is 11.2 g/dL." in hemoglobin_section
    assert INSUFFICIENT_INFORMATION_MESSAGE not in hemoglobin_section


def test_real_template_explanation_without_sources_is_insufficient(audit_logs):
    result = run(audit_logs, retrieval_service=real_retrieval())  # nothing found

    assert result.status == "approved"
    section = result.final_response.split("\n\n")[0]
    assert INSUFFICIENT_INFORMATION_MESSAGE in section
    # The agent's own claim-free insufficient wording is kept.
    assert "There is not enough reliable information" in section
    assert "From \"" not in section


def test_rejected_once_then_regenerated_draft_is_approved(audit_logs):
    explanation = FakeExplanation(UNSAFE_TEXT, SAFE_TEXT)

    result = run(audit_logs, explanation_service=explanation)

    assert result.status == "approved"
    assert UNSAFE_TEXT not in result.final_response
    assert [r.rejection_feedback for r in explanation.requests] == [[], [rejection_note("diagnosis_detected")]]
    assert "diagnosis_detected" in explanation.requests[1].rejection_feedback[0]


def test_rejected_every_attempt_returns_fallback_without_draft(audit_logs):
    explanation = FakeExplanation(UNSAFE_TEXT)

    result = run(audit_logs, explanation_service=explanation)

    assert len(explanation.requests) == 3  # 1 attempt + 2 retries
    # The feedback list grows with each rejection.
    assert [len(r.rejection_feedback) for r in explanation.requests] == [0, 1, 2]
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
    # The finding is still sent, with no sources; the generated text that came back is not used.
    [request] = explanation.requests
    assert [f.test for f in request.findings] == ["Hemoglobin"]
    assert request.retrieved_sources == []
    assert SAFE_TEXT not in result.final_response


def test_all_findings_and_all_sources_are_sent_every_time(audit_logs):
    explanation = FakeExplanation(SAFE_TEXT)

    result = run(
        audit_logs,
        document_service=fake_document(HEMOGLOBIN, PLATELETS_UNVERIFIED),
        explanation_service=explanation,
    )

    [request] = explanation.requests
    assert [(f.test, f.value, f.unit, f.reference_range, f.status) for f in request.findings] == [
        ("Hemoglobin", 11.2, "g/dL", "12.0-15.5", "low"),
        ("Platelets", 250.0, None, "150-400", "normal"),
    ]
    assert [s.test_name for s in request.retrieved_sources] == ["Hemoglobin"]
    assert (request.task_id, request.report_id, request.user_id) == (result.task_id, result.report_id, "user-1")
    hemoglobin_section, platelets_section = result.final_response.split("\n\n")[:2]
    assert SAFE_TEXT in hemoglobin_section
    assert platelets_section.startswith("Platelets:")
    assert INSUFFICIENT_INFORMATION_MESSAGE in platelets_section
    assert SAFE_TEXT not in platelets_section


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


def test_explanation_audit_counts_findings_by_generation_mode(audit_logs):
    # Real template service: Hemoglobin has a source (template), Platelets has none (insufficient).
    result = run(audit_logs, document_service=fake_document(HEMOGLOBIN, PLATELETS_UNVERIFIED))

    [entry] = list(audit_logs.find({"task_id": result.task_id, "agent": "explanation_agent"}))
    assert (entry["action"], entry["status"]) == ("explain", "success")
    assert entry["details"] == {"attempt": 1, "generation_modes": {"insufficient": 1, "template": 1}}


def test_explanation_response_for_another_task_falls_back(audit_logs):
    explanation = FakeExplanation(SAFE_TEXT)

    def wrong_task_explanation(request):
        return explanation(request).model_copy(update={"task_id": "another-task"})

    result = run(audit_logs, explanation_service=wrong_task_explanation)

    assert result.status == "fallback"
    assert result.message == EXPLANATION_UNAVAILABLE_MESSAGE
    assert result.final_response is None


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
    [source] = explanation.requests[0].retrieved_sources
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

    assert [s.test_name for s in explanation.requests[0].retrieved_sources] == ["Hemoglobin"]
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
    assert explanation.requests[0].retrieved_sources == []
    assert INSUFFICIENT_INFORMATION_MESSAGE in result.final_response
    assert SAFE_TEXT not in result.final_response


def test_retrieval_exception_is_audited_without_crashing(audit_logs):
    result = run(audit_logs, retrieval_service=real_retrieval(error=RuntimeError("index unavailable")))

    assert result.status == "approved"
    [entry] = list(audit_logs.find({"task_id": result.task_id, "agent": "retrieval_agent"}))
    assert (entry["action"], entry["status"]) == ("retrieve", "error")
    assert entry["details"] == {"error_type": "RuntimeError"}


def test_default_retrieval_keeps_evidence_when_one_test_fails(monkeypatch):
    from agents.retrieval_models import RetrievalResponse, RetrievalResult
    from agents.vector_store import VectorIndexNotBuiltError

    class Agent:
        def retrieve(self, request):
            if "VLDL Cholesterol" in request.test_names:
                raise VectorIndexNotBuiltError("missing index")
            return RetrievalResponse(**request.model_dump(exclude={"test_names"}),
                                     results=[RetrievalResult(test_name=n, found=False) for n in request.test_names])

    monkeypatch.setattr(coordinator, "_default_retrieval_agent", lambda: Agent())
    request = RetrievalRequest(task_id="t", report_id="r", user_id="u", test_names=["Total Cholesterol", "VLDL Cholesterol"])

    response = coordinator.default_retrieval(request)

    assert [r.test_name for r in response.results] == ["Total Cholesterol", "VLDL Cholesterol"]
    assert (response.task_id, response.report_id, response.user_id) == ("t", "r", "u")
    only_failing = request.model_copy(update={"test_names": ["VLDL Cholesterol", "VLDL Cholesterol"]})
    with pytest.raises(RuntimeError):
        coordinator.default_retrieval(only_failing)
