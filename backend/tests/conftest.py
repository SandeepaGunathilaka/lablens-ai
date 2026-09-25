import os

# Must be set before `main` is imported, because security/tokens.py refuses to load without it.
os.environ["JWT_SECRET_KEY"] = "test-secret-key-not-for-production"

import mongomock  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from database import get_users_collection  # noqa: E402
from main import app  # noqa: E402
from security.auth import ensure_user_indexes  # noqa: E402


@pytest.fixture
def users():
    """A fresh in-memory users collection for every test."""
    collection = mongomock.MongoClient().db.users
    ensure_user_indexes(collection)
    return collection


@pytest.fixture
def client(users):
    app.dependency_overrides[get_users_collection] = lambda: users
    yield TestClient(app)
    app.dependency_overrides.clear()
