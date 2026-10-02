"""Tests for saved reports and chats, including ownership and the safety fallback."""

import pymupdf as fitz
import pytest

from agents.safety_agent import RejectedResponse
from coordinator import FALLBACK_MESSAGE, AnalyzedLabResult, CoordinatorResult

REPORTS_URL = "/api/reports"
CHATS_URL = "/api/chats"


def _text_pdf(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


def _upload(client, headers, content=None, filename="cbc.pdf"):
    content = content if content is not None else _text_pdf("Hemoglobin 11.2 g/dL 12-16")
    return client.post(REPORTS_URL, files={"file": (filename, content, "application/pdf")}, headers=headers)


@pytest.fixture
def report(client, auth_headers):
    response = _upload(client, auth_headers("user-1"))
    assert response.status_code == 201
    return response.json()


def _new_chat(client, headers, report_id):
    response = client.post(CHATS_URL, json={"report_id": report_id}, headers=headers)
    assert response.status_code == 201
    return response.json()


# --- Reports -------------------------------------------------------------------------


def test_reports_require_authentication(client):
    assert client.get(REPORTS_URL).status_code == 401
    assert _upload(client, headers=None).status_code == 401


def test_upload_saves_report_with_results_and_sources(client, auth_headers, report, storage):
    assert report["name"] == "cbc"
    assert report["status"] == "approved"
    assert report["report_type"] == "cbc"
    assert report["has_file"] is True
    assert report["results"][0]["test"] == "Hemoglobin"
    assert report["results"][0]["status"] == "low"
    assert report["final_response"]
    assert report["sources"][0]["test_name"] == "Hemoglobin"
    assert report["sources"][0]["links"][0]["url"].startswith("https://medlineplus.gov/")
    assert storage.report_files.count_documents({"report_id": report["id"]}) == 1

    listed = client.get(REPORTS_URL, headers=auth_headers("user-1")).json()
    assert [(r["id"], r["result_statuses"], r["chat_count"]) for r in listed] == [(report["id"], ["low"], 0)]


def test_failed_analysis_is_not_saved(client, auth_headers, storage, monkeypatch):
    def failed(filename, content, user_id, *, audit_logs):
        return CoordinatorResult(
            task_id="t", report_id="r", user_id=user_id, status="failed", message="The uploaded file is empty."
        )

    monkeypatch.setattr("api.reports.analyze_report", failed)

    response = _upload(client, auth_headers("user-1"), content=b"")

    assert response.status_code == 422
    assert response.json() == {"detail": "The uploaded file is empty."}
    assert storage.reports.count_documents({}) == 0
    assert storage.report_files.count_documents({}) == 0


def test_rename_report(client, auth_headers, report):
    response = client.patch(f"{REPORTS_URL}/{report['id']}", json={"name": "  Annual blood count "}, headers=auth_headers("user-1"))

    assert response.status_code == 200
    assert response.json()["name"] == "Annual blood count"


def test_get_original_file(client, auth_headers, report):
    response = client.get(f"{REPORTS_URL}/{report['id']}/file", headers=auth_headers("user-1"))

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")


def test_delete_file_keeps_report(client, auth_headers, report, storage):
    headers = auth_headers("user-1")
    response = client.delete(f"{REPORTS_URL}/{report['id']}/file", headers=headers)

    assert response.status_code == 200
    assert response.json()["has_file"] is False
    assert response.json()["results"] == report["results"]
    assert storage.report_files.count_documents({}) == 0
    assert client.get(f"{REPORTS_URL}/{report['id']}/file", headers=headers).status_code == 404


def test_delete_report_removes_file_and_chats(client, auth_headers, report, storage):
    headers = auth_headers("user-1")
    _new_chat(client, headers, report["id"])

    response = client.delete(f"{REPORTS_URL}/{report['id']}", headers=headers)

    assert response.status_code == 204
    assert storage.reports.count_documents({}) == 0
    assert storage.report_files.count_documents({}) == 0
    assert storage.chats.count_documents({}) == 0
    assert client.get(f"{REPORTS_URL}/{report['id']}", headers=headers).status_code == 404


def test_other_users_report_looks_unknown(client, auth_headers, report):
    other = auth_headers("user-2")
    rid = report["id"]

    assert client.get(REPORTS_URL, headers=other).json() == []
    assert client.get(f"{REPORTS_URL}/{rid}", headers=other).status_code == 404
    assert client.get(f"{REPORTS_URL}/{rid}/file", headers=other).status_code == 404
    assert client.patch(f"{REPORTS_URL}/{rid}", json={"name": "x"}, headers=other).status_code == 404
    assert client.delete(f"{REPORTS_URL}/{rid}", headers=other).status_code == 404
    assert client.post(CHATS_URL, json={"report_id": rid}, headers=other).status_code == 404


# --- Chats ---------------------------------------------------------------------------


def test_chat_lifecycle(client, auth_headers, report):
    headers = auth_headers("user-1")
    chat = _new_chat(client, headers, report["id"])
    assert chat["title"] == "New chat"
    assert chat["messages"] == []

    response = client.post(
        f"{CHATS_URL}/{chat['id']}/messages", json={"question": "What does hemoglobin measure?"}, headers=headers
    )

    assert response.status_code == 200
    body = response.json()
    assert body["chat"]["title"] == "What does hemoglobin measure?"
    assert body["chat"]["message_count"] == 2
    assert body["user_message"]["text"] == "What does hemoglobin measure?"
    assert body["user_message"]["test_names"] == ["Hemoglobin"]
    answer = body["assistant_message"]["answer"]
    assert answer["status"] == "approved"
    [finding] = answer["findings"]
    assert finding["test"] == "Hemoglobin"
    assert finding["status"] == "low"
    assert finding["sources"][0]["url"].startswith("https://medlineplus.gov/")

    # Continuing the chat later returns the stored conversation.
    saved = client.get(f"{CHATS_URL}/{chat['id']}", headers=headers).json()
    assert [m["role"] for m in saved["messages"]] == ["user", "assistant"]
    assert client.get(f"{CHATS_URL}?report_id={report['id']}", headers=headers).json()[0]["id"] == chat["id"]

    renamed = client.patch(f"{CHATS_URL}/{chat['id']}", json={"title": "Hemoglobin"}, headers=headers)
    assert renamed.json()["title"] == "Hemoglobin"

    assert client.delete(f"{CHATS_URL}/{chat['id']}", headers=headers).status_code == 204
    assert client.get(f"{CHATS_URL}/{chat['id']}", headers=headers).status_code == 404


def test_rejected_answer_returns_fallback_without_generated_text(client, auth_headers, report, audit_logs, monkeypatch):
    monkeypatch.setattr("chat_service.validate_draft", lambda payload: RejectedResponse(reason="diagnosis_detected"))
    headers = auth_headers("user-1")
    chat = _new_chat(client, headers, report["id"])

    response = client.post(f"{CHATS_URL}/{chat['id']}/messages", json={"question": "What does my hemoglobin mean?"}, headers=headers)

    answer = response.json()["assistant_message"]["answer"]
    assert answer == {"task_id": answer["task_id"], "status": "fallback", "findings": [], "message": FALLBACK_MESSAGE}
    safety = list(audit_logs.find({"task_id": answer["task_id"], "agent": "safety_agent"}))
    assert [e["details"]["attempt"] for e in safety] == [1, 2, 3]


@pytest.mark.parametrize("question, status, intent, text", [
    ("Do I have anemia?", "redirect", "diagnosis", "qualified healthcare professional"),
    ("k", "reply", "small_talk", "Ask me about any result"),
    ("What is the capital of France?", "reply", "off_topic", "only answer questions about the lab results"),
])
def test_questions_needing_no_explanation_get_a_direct_reply(client, auth_headers, report, audit_logs, monkeypatch,
                                                             question, status, intent, text):
    monkeypatch.setattr("chat_service.default_explanation", lambda request: pytest.fail("explanation must not run"))
    headers = auth_headers("user-1")
    chat = _new_chat(client, headers, report["id"])

    body = client.post(f"{CHATS_URL}/{chat['id']}/messages", json={"question": question}, headers=headers).json()

    answer = body["assistant_message"]["answer"]
    assert answer["status"] == status
    assert text in answer["message"]
    assert answer["findings"] == []
    assert body["user_message"]["test_names"] == []
    [entry] = audit_logs.find({"task_id": answer["task_id"]})
    assert (entry["status"], entry["details"]) == (status, {"intent": intent})


def test_chat_belongs_to_its_owner(client, auth_headers, report):
    chat = _new_chat(client, auth_headers("user-1"), report["id"])
    other = auth_headers("user-2")

    assert client.get(CHATS_URL, headers=other).json() == []
    assert client.get(f"{CHATS_URL}/{chat['id']}", headers=other).status_code == 404
    assert client.post(f"{CHATS_URL}/{chat['id']}/messages", json={"question": "hi"}, headers=other).status_code == 404
    assert client.delete(f"{CHATS_URL}/{chat['id']}", headers=other).status_code == 404


def test_blank_question_is_rejected(client, auth_headers, report):
    headers = auth_headers("user-1")
    chat = _new_chat(client, headers, report["id"])

    response = client.post(f"{CHATS_URL}/{chat['id']}/messages", json={"question": "   "}, headers=headers)

    assert response.status_code == 422


def test_answer_without_sources_is_marked_insufficient(monkeypatch, audit_logs):
    from chat_service import answer_question

    result = AnalyzedLabResult(test="Ferritin", value=40, unit="ng/mL", reference_range="20-250", confidence=0.99, needs_verification=False, status="normal")

    answer = answer_question(
        report_id="r", user_id="u", results=[result], retrieved_sources=[], question="What is ferritin?", audit_logs=audit_logs
    )

    assert answer.status == "approved"
    [finding] = answer.findings
    assert finding.insufficient_information is True
    assert finding.sources == []
