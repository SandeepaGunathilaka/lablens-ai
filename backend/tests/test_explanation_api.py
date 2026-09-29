import pytest
from fastapi.testclient import TestClient

from explanation_agent.router import clear_rate_limits, get_explanation_service
from explanation_agent.service import ExplanationService
from main import app


@pytest.fixture
def client():
    app.dependency_overrides[get_explanation_service] = lambda: ExplanationService(
        mode="template"
    )
    clear_rate_limits()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    clear_rate_limits()


def test_explanation_endpoint_returns_the_contract(client):
    response = client.post(
        "/explanation",
        json={
            "task_id": "task-api-1",
            "test_name": "LDL",
            "value": "160",
            "unit": "mg/dL",
            "reference_range": "<100",
            "retrieved_sources": [
                {
                    "title": "LDL note",
                    "excerpt": "LDL is a form of cholesterol that carries cholesterol in the blood.",
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "high"
    assert body["generation_mode"] == "template"
    assert body["value"] == "160"
    assert "160" in body["explanation"]
    assert "LDL note" in body["sources_used"]
    assert set(body) >= {
        "what_it_measures",
        "explanation",
        "possible_meaning",
        "recommended_discussion",
        "insufficient_information",
        "status",
    }


def test_explanation_endpoint_rejects_an_empty_test_name(client):
    response = client.post(
        "/explanation",
        json={
            "task_id": "task-api-2",
            "test_name": " ",
            "value": "10",
            "unit": "g/dL",
        },
    )

    assert response.status_code == 422


def test_health_endpoint_still_responds(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
