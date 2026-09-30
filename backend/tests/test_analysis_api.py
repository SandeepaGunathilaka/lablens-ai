"""Tests for the authenticated upload adapter around the Coordinator pipeline."""

from coordinator import CoordinatorResult


ANALYZE_URL = "/api/analyze-report"


def _upload(client, headers=None):
    return client.post(
        ANALYZE_URL,
        files={"file": ("cbc.pdf", b"%PDF-sample", "application/pdf")},
        headers=headers,
    )


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
