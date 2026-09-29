import os

# Must be set before `main` is imported, because security/tokens.py refuses to load without it.
os.environ["JWT_SECRET_KEY"] = "test-secret-key-not-for-production"

import mongomock  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from database import get_audit_logs_collection, get_users_collection  # noqa: E402
from logging_service import ensure_audit_log_indexes  # noqa: E402
from main import app  # noqa: E402
from security.auth import ensure_user_indexes  # noqa: E402


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
def client(users, audit_logs):
    app.dependency_overrides[get_users_collection] = lambda: users
    app.dependency_overrides[get_audit_logs_collection] = lambda: audit_logs
    yield TestClient(app)
    app.dependency_overrides.clear()
