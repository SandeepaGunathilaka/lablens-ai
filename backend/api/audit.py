from fastapi import APIRouter, Depends
from pydantic import BaseModel
from pymongo.collection import Collection

from database import get_audit_logs_collection

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
def get_task_audit_log(task_id: str, audit_logs: Collection = Depends(get_audit_logs_collection)):
    """All audit entries for one task, oldest first. Unknown task ids return []."""
    # {"_id": 0} leaves out MongoDB's internal id, which can't be turned into JSON.
    # _id breaks ties between entries written in the same microsecond, keeping insertion order.
    cursor = audit_logs.find({"task_id": task_id}, {"_id": 0}).sort([("timestamp", 1), ("_id", 1)])
    return list(cursor)
