"""Evidence capture for the Student 4 Information Retrieval and Security assessment (IR-01..IR-15)."""

import hashlib
import os
from importlib import metadata
from pathlib import Path

os.environ["EXPLANATION_PROVIDER"] = "template"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pytest  # noqa: E402

from evidence_support import BACKEND_DIR, base_environment, evidence, pytest_runtest_makereport, write_results  # noqa: E402,F401
from pipeline_support import RealRetrieval  # noqa: E402

SUITE_DIR = Path(__file__).parent


@pytest.fixture(scope="session")
def real_retrieval(tmp_path_factory) -> RealRetrieval:
    """The production keyword + MiniLM/Chroma stack over the curated knowledge base."""
    return RealRetrieval(tmp_path_factory.mktemp("chroma") / "index")


def _environment() -> dict:
    kb = hashlib.sha256()
    for path in sorted((BACKEND_DIR / "data" / "knowledge_base").glob("*.json")):
        kb.update(path.read_bytes())
    return {
        **base_environment(),
        "component": "Medical Retrieval Agent (keyword + semantic hybrid), auth/API layer (FastAPI)",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2 (real model, CPU)",
        "vector_store": f"ChromaDB {metadata.version('chromadb')} (index rebuilt from the curated KB per run)",
        "knowledge_base_sha256": kb.hexdigest()[:16],
        "explanation_provider": "template (deterministic; scripted model where stated)",
        "database": f"mongomock {metadata.version('mongomock')} (in-memory MongoDB)",
    }


def pytest_sessionfinish(session, exitstatus):
    write_results(SUITE_DIR, _environment())
