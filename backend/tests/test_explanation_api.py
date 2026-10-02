import pytest

from explanation_agent.router import clear_rate_limits, get_explanation_service
from explanation_agent.service import ExplanationService
from main import app

URL = "/explanation"


@pytest.fixture(autouse=True)
def template_service():
    app.dependency_overrides[get_explanation_service] = lambda: ExplanationService(mode="template")
    clear_rate_limits()
    yield
    clear_rate_limits()


def payload(user_id="user-1", **overrides):
    body = {
        "task_id": "task-api-1",
        "report_id": "report-api-1",
        "user_id": user_id,
        "findings": [
            {"test": "LDL", "value": 160, "unit": "mg/dL", "reference_range": "<100", "status": "high"},
        ],
        "retrieved_sources": [
            {
                "test_name": "LDL",
                "information": {"passages": ["LDL is a form of cholesterol that carries cholesterol in the blood."]},
                "sources": [{"title": "LDL note", "url": "https://example.test/ldl"}],
            }
        ],
    }
    body.update(overrides)
    return body


def test_explanation_endpoint_returns_the_contract(client, auth_headers):
    response = client.post(URL, json=payload(), headers=auth_headers("user-1"))

    assert response.status_code == 200
    body = response.json()
    assert (body["task_id"], body["report_id"], body["user_id"]) == ("task-api-1", "report-api-1", "user-1")
    [finding] = body["findings"]
    assert finding["test_name"] == "LDL"
    assert finding["result"] == "160 mg/dL"
    assert finding["status"] == "high"
    assert finding["generation_mode"] == "template"
    assert finding["sources_used"] == ["LDL note"]
    assert set(finding) >= {"what_it_measures", "explanation", "possible_meaning", "recommended_discussion"}


def test_explanation_is_written_to_the_audit_log_without_patient_content(client, auth_headers, audit_logs):
    client.post(URL, json=payload(), headers=auth_headers("user-1"))

    [entry] = list(audit_logs.find({}, {"_id": 0}))
    assert (entry["agent"], entry["action"], entry["status"]) == ("explanation_agent", "explain", "success")
    assert (entry["task_id"], entry["report_id"], entry["user_id"]) == ("task-api-1", "report-api-1", "user-1")
    assert entry["details"] == {"finding_count": 1, "generation_modes": {"template": 1}}


def test_no_token_returns_401(client, audit_logs):
    response = client.post(URL, json=payload())

    assert response.status_code == 401
    assert audit_logs.count_documents({}) == 0


def test_another_users_request_returns_403(client, auth_headers, audit_logs):
    response = client.post(URL, json=payload(user_id="user-2"), headers=auth_headers("user-1"))

    assert response.status_code == 403
    assert audit_logs.count_documents({}) == 0


def test_rate_limit_is_per_user(client, auth_headers, monkeypatch):
    monkeypatch.setenv("EXPLANATION_RATE_LIMIT", "1")

    first = client.post(URL, json=payload(), headers=auth_headers("user-1"))
    second = client.post(URL, json=payload(), headers=auth_headers("user-1"))
    other = client.post(URL, json=payload(user_id="user-2"), headers=auth_headers("user-2"))

    assert (first.status_code, second.status_code, other.status_code) == (200, 429, 200)


def test_empty_test_name_is_rejected(client, auth_headers):
    body = payload(findings=[{"test": " ", "value": 10, "unit": "g/dL"}])

    response = client.post(URL, json=body, headers=auth_headers("user-1"))

    assert response.status_code == 422


def test_empty_findings_are_rejected(client, auth_headers):
    response = client.post(URL, json=payload(findings=[]), headers=auth_headers("user-1"))

    assert response.status_code == 422
