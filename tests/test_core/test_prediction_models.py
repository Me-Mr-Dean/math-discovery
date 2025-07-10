from src.core.prediction_models import SimpleLogisticModel


def test_simple_logistic_model():
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    model = SimpleLogisticModel(lr=0.1, epochs=5000)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    accuracy = sum(int(p == t) for p, t in zip(preds, y)) / len(y)
    assert accuracy >= 0.75
