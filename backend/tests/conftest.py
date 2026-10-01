import os

# Must be set before `main` is imported, because security/tokens.py refuses to load without it.
os.environ["JWT_SECRET_KEY"] = "test-secret-key-not-for-production"

import mongomock  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.reports import ensure_report_indexes  # noqa: E402
from database import (  # noqa: E402
    get_audit_logs_collection,
    get_chats_collection,
    get_report_files_collection,
    get_reports_collection,
    get_users_collection,
)
from logging_service import ensure_audit_log_indexes  # noqa: E402
from main import app  # noqa: E402
from security.auth import ensure_user_indexes  # noqa: E402
from security.tokens import create_access_token  # noqa: E402


@pytest.fixture
def users():
    """A fresh in-memory users collection for every test."""
    collection = mongomock.MongoClient().db.users
    ensure_user_indexes(collection)
    return collection


@pytest.fixture
def audit_logs():
    """A fresh in-memory audit_logs collection for every test."""
    collection = mongomock.MongoClient().db.audit_logs
    ensure_audit_log_indexes(collection)
    return collection


@pytest.fixture
def auth_headers():
    """Build an Authorization header with a valid token: auth_headers("user-1")."""

    def make(user_id: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {create_access_token(user_id)}"}

    return make


@pytest.fixture
def storage():
    """Fresh in-memory reports, report_files and chats collections."""
    db = mongomock.MongoClient().db
    ensure_report_indexes(db.reports, db.report_files, db.chats)
    return db


@pytest.fixture
def client(users, audit_logs, storage):
    app.dependency_overrides[get_users_collection] = lambda: users
    app.dependency_overrides[get_audit_logs_collection] = lambda: audit_logs
    app.dependency_overrides[get_reports_collection] = lambda: storage.reports
    app.dependency_overrides[get_report_files_collection] = lambda: storage.report_files
    app.dependency_overrides[get_chats_collection] = lambda: storage.chats
    yield TestClient(app)
    app.dependency_overrides.clear()
