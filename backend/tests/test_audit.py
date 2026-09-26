import logging
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from pymongo.errors import ServerSelectionTimeoutError

from agents.safety_agent import DISCLAIMER
from logging_service import log_event

EVENT = {
    "task_id": "task-1",
    "report_id": "report-1",
    "user_id": "user-1",
    "agent": "safety_agent",
    "action": "validate",
    "status": "approved",
}


# --- log_event --------------------------------------------------------------------


def test_log_event_writes_document_with_all_fields(audit_logs):
    log_event(audit_logs, **EVENT, details={"reason": "none"})

    stored = audit_logs.find_one({"task_id": "task-1"}, {"_id": 0})
    assert set(stored) == {*EVENT, "log_id", "timestamp", "details"}
    assert {k: stored[k] for k in EVENT} == EVENT
    assert stored["details"] == {"reason": "none"}
    uuid.UUID(stored["log_id"])  # raises if it isn't a valid UUID

    timestamp = datetime.fromisoformat(stored["timestamp"])
    assert timestamp.utcoffset() == timezone.utc.utcoffset(None)
    assert abs((datetime.now(timezone.utc) - timestamp).total_seconds()) < 5


def test_log_event_defaults_details_to_empty_dict(audit_logs):
    log_event(audit_logs, **EVENT)

    assert audit_logs.find_one()["details"] == {}


def test_log_event_never_raises_when_database_fails(caplog):
    broken = MagicMock()
    broken.insert_one.side_effect = ServerSelectionTimeoutError("MongoDB is down")

    with caplog.at_level(logging.WARNING, logger="logging_service"):
        log_event(broken, **EVENT)  # must not raise

    assert "Could not write audit log for task task-1" in caplog.text


# --- GET /api/audit/{task_id} -----------------------------------------------------


def test_get_audit_returns_task_entries_sorted_by_timestamp(client, audit_logs):
    def entry(log_id, task_id, timestamp):
        return {**EVENT, "log_id": log_id, "task_id": task_id, "timestamp": timestamp, "details": {}}

    # Inserted out of order, with another task's entry mixed in.
    audit_logs.insert_many([
        entry("second", "task-1", "2026-09-26T10:00:02.000000+00:00"),
        entry("other-task", "task-2", "2026-09-26T10:00:00.000000+00:00"),
        entry("first", "task-1", "2026-09-26T10:00:01.000000+00:00"),
        entry("third", "task-1", "2026-09-26T10:00:03.000000+00:00"),
    ])

    response = client.get("/api/audit/task-1")

    assert response.status_code == 200
    body = response.json()
    assert [e["log_id"] for e in body] == ["first", "second", "third"]
    assert "_id" not in body[0]


def test_get_audit_returns_empty_list_for_unknown_task(client):
    response = client.get("/api/audit/no-such-task")

    assert response.status_code == 200
    assert response.json() == []


# --- Safety Agent integration -----------------------------------------------------


def safety_payload(draft):
    return {
        "task_id": "task-42",
        "report_id": "report-7",
        "user_id": "user-9",
        "original_result": [
            {"test": "Hemoglobin", "value": 11.2, "unit": "g/dL", "reference_range": "12.0-15.5"},
        ],
        "retrieved_sources": [
            {"test_name": "Hemoglobin", "information": {"description": "Hemoglobin carries oxygen."}, "sources": []},
        ],
        "draft_response": draft,
    }


@pytest.mark.parametrize(
    "draft, status, reason",
    [
        (f"Your hemoglobin is 11.2 g/dL. {DISCLAIMER}", "approved", None),
        (f"Your hemoglobin is 11.2 g/dL. You should take iron supplements. {DISCLAIMER}", "rejected", "medication_detected"),
    ],
    ids=["approved", "rejected"],
)
def test_safety_validation_writes_audit_entry(client, audit_logs, draft, status, reason):
    assert client.post("/agents/safety/validate", json=safety_payload(draft)).status_code == 200

    entries = client.get("/api/audit/task-42").json()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["agent"] == "safety_agent"
    assert entry["action"] == "validate"
    assert entry["status"] == status
    assert (entry["report_id"], entry["user_id"]) == ("report-7", "user-9")
    assert entry["details"].get("reason") == reason
    assert set(entry["details"]["checks"]) == {
        "patient_values_verified",
        "diagnosis_detected",
        "medication_detected",
        "unsupported_claim_detected",
        "disclaimer_present",
    }
    # No patient content in the audit trail.
    assert "hemoglobin" not in str(entry).lower()
