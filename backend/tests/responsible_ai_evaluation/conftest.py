"""Evidence capture for the group final report's Responsible AI test suite (section 7.2, RA-01..RA-15)."""

import hashlib
import os
from importlib import metadata
from pathlib import Path

os.environ["EXPLANATION_PROVIDER"] = "template"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from evidence_support import BACKEND_DIR, base_environment, evidence, pytest_runtest_makereport, write_results  # noqa: E402,F401
from pipeline_support import real_retrieval  # noqa: E402,F401

SUITE_DIR = Path(__file__).parent


def _environment() -> dict:
    from explanation_agent.llm import DEFAULT_MODEL
    from explanation_agent.prompt import SYSTEM_PROMPT

    kb = hashlib.sha256()
    for path in sorted((BACKEND_DIR / "data" / "knowledge_base").glob("*.json")):
        kb.update(path.read_bytes())
    return {
        **base_environment(),
        "system_under_test": "Full LabLens pipeline over HTTP: auth -> upload -> Document Agent -> Retrieval -> "
                             "Explanation -> Safety -> saved report -> follow-up chat",
        "explanation_provider": f"template provider; scripted stand-in for Gemini ({DEFAULT_MODEL}) where stated - no network calls",
        "retrieval": "real keyword + all-MiniLM-L6-v2/Chroma hybrid retriever (index rebuilt per run)",
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:16],
        "knowledge_base_sha256": kb.hexdigest()[:16],
        "database": f"mongomock {metadata.version('mongomock')} (in-memory MongoDB)",
    }


def pytest_sessionfinish(session, exitstatus):
    write_results(SUITE_DIR, _environment())
