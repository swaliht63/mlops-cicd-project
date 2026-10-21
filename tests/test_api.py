from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from src.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint():
    data = load_breast_cancer(as_frame=True)
    features = data.data.iloc[0].to_dict()

    response = client.post("/predict", json={"features": features})

    assert response.status_code == 200
    result = response.json()
    assert result["class_name"] in {"malignant", "benign"}
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_rejects_missing_features():
    response = client.post("/predict", json={"features": {"mean radius": 17.99}})

    assert response.status_code == 400
    assert "missing_features" in response.json()["detail"]
