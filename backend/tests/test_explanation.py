"""Legacy explanation HTTP contract tests; no live model calls or credentials."""
from unittest.mock import Mock

import pytest

from api.explanation import get_explanation_service
from agents.explanation.service import ExplanationService
from main import app


@pytest.fixture
def explanation_service():
    service = Mock(spec=ExplanationService)
    service.generate_explanation.return_value = "Educational explanation from the test service."
    app.dependency_overrides[get_explanation_service] = lambda: service
    try:
        yield service
    finally:
        app.dependency_overrides.pop(get_explanation_service, None)


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "LabLens AI backend is running"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_explanation_validation(client, explanation_service):
    payload = {"findings": [{
        "test_name": "Hemoglobin", "value": "11.2", "unit": "g/dL",
        "reference_range": "12.0-15.5", "status": "low",
    }]}
    response = client.post("/explanation/explain", json=payload)
    assert response.status_code == 200
    assert response.json() == {"explanation": explanation_service.generate_explanation.return_value}
    explanation_service.generate_explanation.assert_called_once()
    request = explanation_service.generate_explanation.call_args.args[0]
    assert request.findings[0].model_dump() == payload["findings"][0]


def test_explanation_rejects_invalid_payload(client, explanation_service):
    response = client.post("/explanation/explain", json={"findings": [{}]})
    assert response.status_code == 422
    explanation_service.generate_explanation.assert_not_called()


def test_explanation_service_failure_is_not_success(client, explanation_service):
    explanation_service.generate_explanation.side_effect = RuntimeError("Model unavailable")
    response = client.post("/explanation/explain", json={
        "findings": [{"test_name": "Hemoglobin", "value": "11.2"}],
    })
    assert response.status_code == 500


def test_explanation_dependency_is_lazy_and_cached(monkeypatch):
    from api import explanation

    explanation.get_explanation_service.cache_clear()
    factory = Mock(spec=ExplanationService)
    monkeypatch.setattr(explanation, "ExplanationService", factory)
    try:
        factory.assert_not_called()
        assert explanation.get_explanation_service() is factory.return_value
        assert explanation.get_explanation_service() is factory.return_value
        factory.assert_called_once_with()
    finally:
        explanation.get_explanation_service.cache_clear()
