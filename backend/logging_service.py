import logging
import uuid
from datetime import datetime, timezone
from typing import Literal

from pymongo.collection import Collection

logger = logging.getLogger(__name__)

AuditStatus = Literal["approved", "rejected", "success", "error"]


def ensure_audit_log_indexes(audit_logs: Collection) -> None:
    """Index for GET /api/audit/{task_id}: find by task, already sorted by time."""
    audit_logs.create_index([("task_id", 1), ("timestamp", 1)])


def log_event(
    audit_logs: Collection,
    *,
    task_id: str,
    report_id: str,
    user_id: str,
    agent: str,
    action: str,
    status: AuditStatus,
    details: dict | None = None,
) -> None:
    """Write one audit log entry. Never raises: a failed write only logs a warning.

    Keep patient content (lab values, draft text) out of `details`; store ids,
    decisions and reasons only.
    """
    entry = {
        "log_id": str(uuid.uuid4()),
        "task_id": task_id,
        "report_id": report_id,
        "user_id": user_id,
        "agent": agent,
        "action": action,
        "status": status,
        # Fixed-width ISO-8601 UTC (always includes microseconds), so sorting these
        # strings alphabetically also sorts them chronologically.
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "details": dict(details or {}),
    }
    try:
        audit_logs.insert_one(entry)
    except Exception as exc:  # noqa: BLE001
        # Broad on purpose: besides database errors, a `details` value MongoDB can't
        # store raises a non-PyMongoError, and audit logging must never crash an agent.
        logger.warning("Could not write audit log for task %s (%s/%s): %s", task_id, agent, action, exc)
