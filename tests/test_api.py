from typing import Any, cast

from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from src.api import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint():
    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["model_version"] != "unknown"


def test_predict_endpoint():
    data = cast(Any, load_breast_cancer(as_frame=True))
    features = data.data.iloc[0].to_dict()

    with TestClient(app) as client:
        response = client.post("/predict", json={"features": features})

    assert response.status_code == 200
    result = response.json()
    assert result["class_name"] in {"malignant", "benign"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["model_version"] != "unknown"


def test_predict_rejects_missing_features():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"features": {"mean radius": 17.99}},
        )

    assert response.status_code == 422
    assert "missing_features" in response.json()["detail"]
