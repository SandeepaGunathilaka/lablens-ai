import pytest

from agents.safety_agent import DISCLAIMER

SAFETY_URL = "/agents/safety/validate"


def safety_payload(user_id):
    return {
        "task_id": "task-1",
        "report_id": "report-1",
        "user_id": user_id,
        "original_result": [
            {"test": "Hemoglobin", "value": 11.2, "unit": "g/dL", "reference_range": "12.0-15.5"},
        ],
        "retrieved_sources": [
            {"test_name": "Hemoglobin", "information": {"description": "Hemoglobin carries oxygen."}, "sources": []},
        ],
        "draft_response": f"Your hemoglobin is 11.2 g/dL. {DISCLAIMER}",
    }


def audit_entry(log_id, task_id, user_id):
    return {
        "log_id": log_id,
        "task_id": task_id,
        "report_id": "report-1",
        "user_id": user_id,
        "agent": "safety_agent",
        "action": "validate",
        "status": "approved",
        "timestamp": f"2026-09-29T10:00:0{log_id[-1]}.000000+00:00",
        "details": {},
    }


def call(client, route, headers=None):
    if route == "safety":
        return client.post(SAFETY_URL, json=safety_payload("user-1"), headers=headers)
    return client.get("/api/audit/task-1", headers=headers)


# --- 401: missing or invalid token --------------------------------------------------


@pytest.mark.parametrize("route", ["safety", "audit"])
def test_no_token_returns_401(client, audit_logs, route):
    response = call(client, route)

    assert response.status_code == 401
    assert audit_logs.count_documents({}) == 0


@pytest.mark.parametrize("route", ["safety", "audit"])
def test_garbage_token_returns_401(client, audit_logs, route):
    response = call(client, route, headers={"Authorization": "Bearer not-a-real-token"})

    assert response.status_code == 401
    assert audit_logs.count_documents({}) == 0


# --- POST /agents/safety/validate ----------------------------------------------------


def test_safety_validate_with_own_user_id_works(client, auth_headers):
    response = client.post(SAFETY_URL, json=safety_payload("user-1"), headers=auth_headers("user-1"))

    assert response.status_code == 200
    assert response.json()["approved"] is True


def test_safety_validate_for_another_user_returns_403(client, audit_logs, auth_headers):
    response = client.post(SAFETY_URL, json=safety_payload("user-2"), headers=auth_headers("user-1"))

    assert response.status_code == 403
    # Rejected before anything runs, so nothing is logged under the other user's id.
    assert audit_logs.count_documents({}) == 0


# --- GET /api/audit/{task_id} ---------------------------------------------------------


def test_audit_for_own_task_returns_entries(client, audit_logs, auth_headers):
    audit_logs.insert_many([audit_entry("log-1", "task-1", "user-1"), audit_entry("log-2", "task-1", "user-1")])

    response = client.get("/api/audit/task-1", headers=auth_headers("user-1"))

    assert response.status_code == 200
    assert [e["log_id"] for e in response.json()] == ["log-1", "log-2"]


def test_audit_for_another_users_task_returns_empty_list(client, audit_logs, auth_headers):
    audit_logs.insert_one(audit_entry("log-1", "task-1", "user-2"))

    other_users_task = client.get("/api/audit/task-1", headers=auth_headers("user-1"))
    unknown_task = client.get("/api/audit/no-such-task", headers=auth_headers("user-1"))

    # Identical answers, so the caller can't tell that user-2's task exists.
    assert other_users_task.status_code == unknown_task.status_code == 200
    assert other_users_task.json() == unknown_task.json() == []


def test_audit_only_returns_callers_own_entries(client, audit_logs, auth_headers):
    audit_logs.insert_many([audit_entry("log-1", "task-1", "user-1"), audit_entry("log-2", "task-1", "user-2")])

    response = client.get("/api/audit/task-1", headers=auth_headers("user-1"))

    assert [e["log_id"] for e in response.json()] == ["log-1"]
