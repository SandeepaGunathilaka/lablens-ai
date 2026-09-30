"""Tests for the authenticated upload adapter around the Coordinator pipeline."""

import pymupdf as fitz

from coordinator import CoordinatorResult


ANALYZE_URL = "/api/analyze-report"


def _upload(client, headers=None):
    return client.post(
        ANALYZE_URL,
        files={"file": ("cbc.pdf", b"%PDF-sample", "application/pdf")},
        headers=headers,
    )


def _text_pdf(text: str) -> bytes:
    """Create a selectable-text PDF for the real upload-pipeline test."""
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


def test_analyze_report_requires_authentication(client):
    response = _upload(client)

    assert response.status_code == 401


def test_analyze_report_forwards_upload_and_authenticated_user(client, auth_headers, monkeypatch):
    captured = {}

    def fake_analyze_report(filename, content, user_id, *, audit_logs):
        captured.update(
            filename=filename,
            content=content,
            user_id=user_id,
            audit_logs=audit_logs,
        )
        return CoordinatorResult(
            task_id="task-1",
            report_id="report-1",
            user_id=user_id,
            status="approved",
            report_type="cbc",
            results=[],
            final_response="Approved response.",
        )

    monkeypatch.setattr("api.analysis.analyze_report", fake_analyze_report)

    response = _upload(client, auth_headers("user-1"))

    assert response.status_code == 200
    assert captured["filename"] == "cbc.pdf"
    assert captured["content"] == b"%PDF-sample"
    assert captured["user_id"] == "user-1"
    assert captured["audit_logs"] is not None
    assert response.json() == {
        "task_id": "task-1",
        "report_id": "report-1",
        "user_id": "user-1",
        "status": "approved",
        "report_type": "cbc",
        "results": [],
        "final_response": "Approved response.",
        "message": None,
    }


def test_analyze_report_runs_real_pipeline_over_http(client, auth_headers, audit_logs):
    """Exercise HTTP upload through Document, Retrieval, placeholder Explanation and Safety.

    Hemoglobin is in the curated keyword knowledge base, so this uses the real
    Retrieval Agent without requiring a network model download or Chroma search.
    """
    report = _text_pdf("Hemoglobin 11.2 g/dL 12-16")

    response = client.post(
        ANALYZE_URL,
        files={"file": ("cbc.pdf", report, "application/pdf")},
        headers=auth_headers("user-1"),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "user-1"
    assert payload["status"] == "approved"
    assert payload["report_type"] == "cbc"
    assert payload["results"] == [
        {
            "test": "Hemoglobin",
            "value": 11.2,
            "unit": "g/dL",
            "reference_range": "12-16",
            "confidence": 0.99,
            "needs_verification": False,
            "status": "low",
            "warning": None,
        }
    ]
    assert payload["final_response"]
    assert payload["message"] is None

    entries = list(audit_logs.find({"task_id": payload["task_id"]}, {"_id": 0}))
    assert [(entry["agent"], entry["status"]) for entry in entries] == [
        ("document_agent", "success"),
        ("coordinator", "success"),
        ("retrieval_agent", "success"),
        ("explanation_agent", "success"),
        ("safety_agent", "approved"),
        ("coordinator", "approved"),
    ]
