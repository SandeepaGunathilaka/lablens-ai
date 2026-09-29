from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "LabLens AI backend is running"
    )


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_explanation_validation():

    payload = {
        "findings": [
            {
                "test_name": "Hemoglobin",
                "value": "11.2",
                "unit": "g/dL",
                "reference_range": "12.0-15.5",
                "status": "low"
            }
        ]
    }

    response = client.post(
        "/explanation/explain",
        json=payload
    )

    assert response.status_code == 200
    assert "explanation" in response.json()