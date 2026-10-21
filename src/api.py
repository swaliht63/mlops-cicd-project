from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "artifacts" / "model.joblib"

app = FastAPI(title="Breast Cancer Model API")


class PredictionRequest(BaseModel):
    features: dict[str, float]


def load_model_bundle():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run `python -m src.train` first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    bundle = load_model_bundle()
    expected_features = bundle["feature_names"]

    missing = [name for name in expected_features if name not in request.features]
    extra = [name for name in request.features if name not in expected_features]

    if missing or extra:
        raise HTTPException(
            status_code=400,
            detail={"missing_features": missing, "unexpected_features": extra},
        )

    row = pd.DataFrame(
        [[request.features[name] for name in expected_features]],
        columns=expected_features,
    )
    prediction = int(bundle["model"].predict(row)[0])
    probability = float(bundle["model"].predict_proba(row)[0].max())

    return {
        "prediction": prediction,
        "class_name": bundle["target_names"][prediction],
        "confidence": probability,
    }
