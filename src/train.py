from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
from datetime import datetime, timezone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = MODEL_DIR / "model.joblib"


def train_model():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
    "malignant_precision": float(
        precision_score(y_test, predictions, pos_label=0, zero_division=0)
    ),
    "malignant_recall": float(
        recall_score(y_test, predictions, pos_label=0, zero_division=0)
    ),
    "malignant_f1": float(
        f1_score(y_test, predictions, pos_label=0, zero_division=0)
    ),
    }
    accuracy = metrics["accuracy"]


    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metrics(metrics)


    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "feature_names": list(X.columns),
            "target_names": list(data.target_names),
            "accuracy": accuracy,
            "metrics": metrics,

        },
        MODEL_PATH,
    )

    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    print(f"Saved model to: {MODEL_PATH}")

    return accuracy


if __name__ == "__main__":
    train_model()
