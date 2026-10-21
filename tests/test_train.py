from src.train import train_model


def test_training_creates_model():
    accuracy = train_model()
    assert 0.8 <= accuracy <= 1.0
