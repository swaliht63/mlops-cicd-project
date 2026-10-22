from src.train import train_model


def test_training_meets_quality_thresholds():
    train_model()

    from pathlib import Path
    import joblib

    model_path = Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"
    bundle = joblib.load(model_path)
    metrics = bundle["metrics"]

    assert metrics["accuracy"] >= 0.90
    assert metrics["balanced_accuracy"] >= 0.90
    assert metrics["malignant_precision"] >= 0.90
    assert metrics["malignant_recall"] >= 0.90
    assert metrics["malignant_f1"] >= 0.90
