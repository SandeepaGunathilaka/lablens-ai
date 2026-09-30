"""End-to-end test of POST /api/analyze-report through the real Coordinator path.

Why this exists alongside tests/test_analysis_api.py:
    test_analysis_api.py replaces analyze_report() entirely, so it only proves the
    route's wiring (authentication, upload handling, forwarding the user id). This
    file runs the actual pipeline behind the same route, to check what the fake
    there can't:

    - The full audit trail in detail: every stage writes its entry, in order, all
      under the same task_id, report_id and user_id, with nothing logged elsewhere.
    - The Safety Agent is really in the loop, not bypassed. The explanation stub's
      text is validated by the real Safety Agent, so an unsafe stub makes the main
      test fail. (Verified by swapping in an unsupported claim, which turned the
      result into "fallback".) It is not only a check that the happy path returns 200.
    - Both retrieval paths through HTTP: a test found in the knowledge base gets an
      explanation; one that isn't gets the insufficient-information message.

    HybridRetriever is faked (no ChromaDB, no embedding model), which keeps the
    test fast and deterministic, so its assertions can be exact.

Real: authentication, upload handling, Document Agent (selectable-text PDF, so no
Tesseract), MedicalRetrievalAgent, Safety Agent and audit logging (on mongomock).
Faked: only HybridRetriever and the Explanation service, which is intentionally not
real yet.
"""

import pymupdf as fitz
import pytest

import coordinator
from agents.hybrid_retriever import HybridRetrievalResult
from agents.knowledge_base import KnowledgeDocument, KnowledgeSource
from agents.retrieval_agent import MedicalRetrievalAgent
from agents.safety_agent import DISCLAIMER
from coordinator import INSUFFICIENT_INFORMATION_MESSAGE, ExplainedFinding, ExplanationOutput

ANALYZE_URL = "/api/analyze-report"

# Synthetic report, generated in memory (no patient data is committed; see
# tests/fixtures/document_agent/README.md). Hemoglobin is in the fake knowledge
# base, Platelets is not.
REPORT_TEXT = "Complete Blood Count\nHemoglobin 11.2 g/dL 12.0-15.5\nPlatelets 250 x10^3/uL 150-400"

HEMOGLOBIN_DOCUMENT = KnowledgeDocument(
    id="hemoglobin",
    test_name="Hemoglobin",
    aliases=["Hgb"],
    report_type="cbc",
    title="Hemoglobin test",
    definition="A blood test.",
    what_it_measures="Hemoglobin is a protein in red blood cells that carries oxygen.",
    general_information="Results are compared with a reference range.",
    source=KnowledgeSource(
        publisher="MedlinePlus",
        title="Hemoglobin Test",
        url="https://medlineplus.gov/lab-tests/hemoglobin-test/",
        accessed_date="2026-09-01",
    ),
)

STUB_MARKER = "[explanation stub]"


def make_pdf(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


class FakeHybridRetriever:
    """Stands in for HybridRetriever: finds only the given documents."""

    def __init__(self, *documents):
        self.documents = {d.test_name.lower(): d for d in documents}
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        document = self.documents.get(query.lower())
        if document is None:
            return HybridRetrievalResult(None, False, "none", None)
        return HybridRetrievalResult(document, True, "keyword", None)


class StubExplanation:
    """Deterministic, clearly fake explanation text with every ExplainedFinding field filled."""

    def __init__(self):
        self.calls = []

    def __call__(self, findings, retrieved_sources, instruction):
        self.calls.append({"tests": [f.test for f in findings], "instruction": instruction})
        return ExplanationOutput(
            findings=[
                ExplainedFinding(
                    test_name=f.test,
                    result=str(f.value),
                    status=f.status,
                    what_it_measures=f"{STUB_MARKER} {f.test} is a protein in red blood cells that carries oxygen.",
                    explanation=f"{STUB_MARKER} This is placeholder explanation text for testing.",
                    possible_meaning=f"{STUB_MARKER} A result outside the range on your report is worth asking about.",
                    recommended_discussion=f"{STUB_MARKER} Ask your doctor what this result means for you.",
                )
                for f in findings
            ]
        )


@pytest.fixture
def hybrid(monkeypatch):
    """Route the Coordinator's real default_retrieval to a real agent over a fake hybrid retriever."""
    fake = FakeHybridRetriever(HEMOGLOBIN_DOCUMENT)
    monkeypatch.setattr(coordinator, "_default_retrieval_agent", lambda: MedicalRetrievalAgent(fake))
    return fake


@pytest.fixture
def explanation(monkeypatch):
    stub = StubExplanation()
    monkeypatch.setattr(coordinator, "placeholder_explanation", stub)
    return stub


def test_analyze_report_runs_real_pipeline_end_to_end(client, auth_headers, audit_logs, hybrid, explanation):
    response = client.post(
        ANALYZE_URL,
        files={"file": ("cbc.pdf", make_pdf(REPORT_TEXT), "application/pdf")},
        headers=auth_headers("user-1"),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approved"
    assert body["user_id"] == "user-1"
    assert body["report_type"] == "cbc"

    # Real Document Agent extraction plus code-computed status.
    assert [(r["test"], r["value"], r["status"]) for r in body["results"]] == [
        ("Hemoglobin", 11.2, "low"),
        ("Platelets", 250.0, "normal"),
    ]

    # Real retrieval asked about both tests; only the found one was sent for explanation.
    assert hybrid.queries == ["Hemoglobin", "Platelets"]
    assert explanation.calls == [{"tests": ["Hemoglobin"], "instruction": None}]

    final = body["final_response"]
    assert final.endswith(DISCLAIMER)
    hemoglobin_section, platelets_section = final.split("\n\n")[:2]
    # Found path: the stub's text made it through the real Safety Agent.
    assert hemoglobin_section.startswith("Hemoglobin:")
    assert STUB_MARKER in hemoglobin_section
    assert INSUFFICIENT_INFORMATION_MESSAGE not in hemoglobin_section
    # Not-found path: no explanation, just the insufficient-information message.
    assert platelets_section.startswith("Platelets:")
    assert INSUFFICIENT_INFORMATION_MESSAGE in platelets_section
    assert STUB_MARKER not in platelets_section

    # One audit trail, tied to this task, covering every stage.
    entries = list(audit_logs.find({"task_id": body["task_id"]}, {"_id": 0}).sort([("timestamp", 1), ("_id", 1)]))
    assert [(e["agent"], e["action"], e["status"]) for e in entries] == [
        ("document_agent", "extract", "success"),
        ("coordinator", "compute_status", "success"),
        ("retrieval_agent", "retrieve", "success"),
        ("explanation_agent", "explain", "success"),
        ("safety_agent", "validate", "approved"),
        ("coordinator", "analyze_report", "approved"),
    ]
    assert {(e["report_id"], e["user_id"]) for e in entries} == {(body["report_id"], "user-1")}
    assert entries[2]["details"] == {"tests_requested": 2, "tests_with_sources": 1}
    assert audit_logs.count_documents({}) == len(entries)  # nothing logged under another task


def test_analyze_report_without_token_runs_nothing(client, audit_logs, hybrid, explanation):
    response = client.post(ANALYZE_URL, files={"file": ("cbc.pdf", make_pdf(REPORT_TEXT), "application/pdf")})

    assert response.status_code == 401
    assert hybrid.queries == []
    assert explanation.calls == []
    assert audit_logs.count_documents({}) == 0
