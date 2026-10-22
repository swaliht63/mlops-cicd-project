from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "artifacts" / "model.joblib"


def load_model_bundle():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)

    required_keys = {"model", "feature_names", "target_names", "accuracy"}
    if not required_keys.issubset(bundle):
        raise RuntimeError("Model artifact is missing required metadata")

    if not bundle["feature_names"] or not bundle["target_names"]:
        raise RuntimeError("Model artifact has empty feature or target metadata")

    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup; fail startup if the artifact is missing or invalid.
    app.state.model_bundle = load_model_bundle()
    yield


app = FastAPI(
    title="Breast Cancer Model API — educational demo",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float] = Field(min_length=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request):
    # The lifespan hook must have loaded a valid model before this is ready.
    return {
        "status": "ready",
        "model_version": request.app.state.model_bundle.get("model_version", "unknown"),
    }


@app.post("/predict")
def predict(request: PredictionRequest, http_request: Request ):
    bundle = http_request.app.state.model_bundle
    expected_features = bundle["feature_names"]

    missing = [name for name in expected_features if name not in request.features]
    extra = [name for name in request.features if name not in expected_features]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unexpected_features": extra},
         )

    try:
        values = [request.features[name] for name in expected_features]
        row = pd.DataFrame([values], columns=expected_features)
        prediction = int(bundle["model"].predict(row)[0])
        probabilities = bundle["model"].predict_proba(row)[0]
        confidence = float(probabilities.max())
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid feature values") from exc

    return {
        "prediction": prediction,
        "class_name": bundle["target_names"][prediction],
        "confidence": confidence,
        "model_version": bundle.get("model_version", "unknown"),
        "disclaimer": "Educational demo only; not for medical use.",
    }
