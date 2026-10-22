from src.train import train_model


def test_training_creates_model_and_meets_quality_threshold():
    accuracy = train_model()
    assert accuracy >= 0.90
