"""Evidence capture for the Student 2 Privacy and Data Leakage assessment (PRIV-01..PRIV-15)."""

import os
from importlib import metadata
from pathlib import Path

os.environ["EXPLANATION_PROVIDER"] = "template"

from evidence_support import base_environment, evidence, pytest_runtest_makereport, write_results  # noqa: E402,F401

SUITE_DIR = Path(__file__).parent


def _environment() -> dict:
    from main import app
    from security.tokens import JWT_ALGORITHM, JWT_EXPIRE_MINUTES

    return {
        **base_environment(),
        "component": "Authentication, authorization, audit logging, saved reports/chats (FastAPI backend) and the React frontend",
        "backend_version": f"{app.title} {app.version} (FastAPI {metadata.version('fastapi')})",
        "auth": f"JWT {JWT_ALGORITHM}, expiry {JWT_EXPIRE_MINUTES} min (python-jose {metadata.version('python-jose')}); "
                f"bcrypt via passlib {metadata.version('passlib')}",
        "database": f"mongomock {metadata.version('mongomock')} (in-memory MongoDB; Atlas itself not contacted)",
        "explanation_provider": "template (deterministic, no external model calls)",
        "browser": "Microsoft Edge via Playwright (frontend served by Vite dev server)",
    }


def pytest_sessionfinish(session, exitstatus):
    write_results(SUITE_DIR, _environment())
