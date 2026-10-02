from fastapi import APIRouter, Depends
from pydantic import BaseModel
from pymongo.collection import Collection

from database import get_audit_logs_collection
from security.dependencies import get_current_user

router = APIRouter(prefix="/api/audit", tags=["audit"])


class AuditLogEntry(BaseModel):
    log_id: str
    task_id: str
    report_id: str
    user_id: str
    agent: str
    action: str
    status: str
    timestamp: str
    details: dict


@router.get("/{task_id}", response_model=list[AuditLogEntry])
def get_task_audit_log(
    task_id: str,
    audit_logs: Collection = Depends(get_audit_logs_collection),
    current_user: str = Depends(get_current_user),
):
    """The caller's own audit entries for one task, oldest first.

    Unknown tasks and other users' tasks both return [], so the response never
    reveals whether someone else's task exists.
    """
    # {"_id": 0} leaves out MongoDB's internal id, which can't be turned into JSON.
    # _id breaks ties between entries written in the same microsecond, keeping insertion order.
    query = {"task_id": task_id, "user_id": current_user}
    cursor = audit_logs.find(query, {"_id": 0}).sort([("timestamp", 1), ("_id", 1)])
    return list(cursor)
