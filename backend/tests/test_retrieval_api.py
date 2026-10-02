"""HTTP contract tests with an injected agent, never a model or vector index."""

import os
import subprocess
import sys
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from api import retrieval
from agents.retrieval_agent import MedicalRetrievalAgent
from agents.retrieval_models import RetrievalRequest, RetrievalResponse
from agents.knowledge_base import KnowledgeBaseValidationError
from agents.semantic_retriever import SemanticRetrievalIntegrityError
from agents.vector_store import VectorIndexNotBuiltError, VectorIndexStaleError
from main import app
from security.tokens import create_access_token


IDS = {"task_id": " task-1 ", "report_id": "report/2", "user_id": "user-3"}
HEADERS = {"Authorization": f"Bearer {create_access_token('user-3')}"}
FOUND = {
    "test_name": "Hgb", "found": True,
    "matches": [{
        "information": "Curated evidence supplied by the mocked agent.",
        "sources": [{"title": "Hemoglobin Test", "url": "https://medlineplus.gov/lab-tests/hemoglobin-test/"}],
    }],
}
MISSING = {"test_name": "unknown", "found": False, "matches": []}


@pytest.fixture(autouse=True)
def no_retrieval_infrastructure(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("API tests must not initialize retrieval infrastructure")

    for target in (
        "agents.embedding_service.EmbeddingService.__init__",
        "agents.vector_store.ChromaVectorStore.__init__",
        "agents.hybrid_retriever.HybridRetriever.__init__",
    ):
        monkeypatch.setattr(target, forbidden)


@pytest.fixture
def agent(client):
    mock = Mock(spec_set=MedicalRetrievalAgent)
    app.dependency_overrides[retrieval.get_retrieval_agent] = lambda: mock
    return mock


@pytest.mark.parametrize("results", [[FOUND], [MISSING], [FOUND, MISSING], [FOUND, MISSING, FOUND]])
def test_success_and_abstention_are_unwrapped_200(client, agent, results, users, audit_logs):
    payload = {**IDS, "test_names": [result["test_name"] for result in results]}
    expected = {**IDS, "results": results}
    agent.retrieve.return_value = RetrievalResponse.model_validate(expected)
    response = client.post("/api/retrieval", json=payload, headers=HEADERS)
    assert response.status_code == 200
    assert response.json() == expected
    agent.retrieve.assert_called_once()
    request = agent.retrieve.call_args.args[0]
    assert isinstance(request, RetrievalRequest)
    assert request.model_dump() == payload
    assert users.count_documents({}) == 0
    assert audit_logs.count_documents({}) == 0


def test_retrieval_requires_login(client, agent):
    response = client.post("/api/retrieval", json={**IDS, "test_names": ["Hgb"]})
    assert response.status_code == 401
    agent.retrieve.assert_not_called()


def test_retrieval_rejects_another_users_id(client, agent):
    response = client.post("/api/retrieval", json={**IDS, "user_id": "user-4", "test_names": ["Hgb"]}, headers=HEADERS)
    assert response.status_code == 403
    agent.retrieve.assert_not_called()


@pytest.mark.parametrize("missing", ["task_id", "report_id", "user_id", "test_names"])
def test_missing_fields_return_422(client, agent, missing):
    payload = {**IDS, "test_names": ["Hgb"]}
    del payload[missing]
    response = client.post("/api/retrieval", json=payload, headers=HEADERS)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", missing]
    agent.retrieve.assert_not_called()


@pytest.mark.parametrize("names", [[], [""], [" \t"], [None], "Hgb"])
def test_invalid_names_return_422(client, agent, names):
    response = client.post("/api/retrieval", json={**IDS, "test_names": names}, headers=HEADERS)
    assert response.status_code == 422
    agent.retrieve.assert_not_called()


@pytest.mark.parametrize("error_type", [
    RuntimeError, VectorIndexNotBuiltError, VectorIndexStaleError,
    KnowledgeBaseValidationError, SemanticRetrievalIntegrityError,
])
def test_internal_errors_are_generic_500(agent, error_type):
    secret = r"C:\private\patient-index user-3 report/2 traceback-sensitive-details"
    agent.retrieve.side_effect = (
        error_type([secret]) if error_type is KnowledgeBaseValidationError else error_type(secret)
    )
    # TestClient normally re-raises server errors; disable that to inspect the wire response.
    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/retrieval", json={**IDS, "test_names": ["Hgb"]}, headers=HEADERS,
    )
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert secret not in response.text
    assert "Traceback" not in response.text
    agent.retrieve.assert_called_once()


def test_provider_initialization_failure_is_also_generic_500(client):
    def failing_provider():
        raise RuntimeError(r"C:\private\model-cache")

    app.dependency_overrides[retrieval.get_retrieval_agent] = failing_provider
    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/retrieval", json={**IDS, "test_names": ["Hgb"]}, headers=HEADERS,
    )
    assert response.status_code == 500
    assert response.text == "Internal Server Error"


def test_provider_is_lazy_and_reuses_agent(monkeypatch):
    retrieval.get_retrieval_agent.cache_clear()
    factory = Mock(spec=MedicalRetrievalAgent)
    monkeypatch.setattr(retrieval, "MedicalRetrievalAgent", factory)
    try:
        factory.assert_not_called()
        first = retrieval.get_retrieval_agent()
        assert retrieval.get_retrieval_agent() is first
        factory.assert_called_once_with()
    finally:
        retrieval.get_retrieval_agent.cache_clear()


def test_fresh_router_import_does_not_load_models_or_chroma():
    script = """
import importlib.abc
import sys

class RejectHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sentence_transformers', 'transformers', 'torch', 'chromadb', 'huggingface_hub'}:
            raise AssertionError('Heavy import: ' + fullname)

sys.meta_path.insert(0, RejectHeavyImports())
from api.retrieval import get_retrieval_agent
assert get_retrieval_agent.cache_info().currsize == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_openapi_uses_existing_contracts(client):
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/api/retrieval"]["post"]
    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/RetrievalRequest",
    }
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/RetrievalResponse",
    }
    assert "422" in operation["responses"]


def test_existing_health_and_root_endpoints(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/").json() == {"message": "LabLens AI backend is running"}
