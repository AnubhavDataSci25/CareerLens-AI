import os
from sklearn.dummy import DummyClassifier
import numpy as np

from src.utils import evaluate_models, save_objects, load_objects


def make_simple_data():
    # simple binary classification
    X_train = np.array([[0], [1], [0], [1]])
    y_train = np.array([0, 1, 0, 1])
    X_test = np.array([[0], [1]])
    y_test = np.array([0, 1])
    return X_train, y_train, X_test, y_test


def test_evaluate_models_returns_report():
    X_train, y_train, X_test, y_test = make_simple_data()
    models = {"dummy": DummyClassifier(strategy="most_frequent")}
    report = evaluate_models(X_train, y_train, X_test, y_test, models)
    assert isinstance(report, dict)
    assert "dummy" in report
    assert 0.0 <= report["dummy"] <= 1.0


def test_save_and_load_objects(tmp_path):
    obj = {"a": 1, "b": "test"}
    file_path = tmp_path / "obj.pkl"
    # save_objects expects a string path
    save_objects(str(file_path), obj)
    loaded = load_objects(str(file_path))
    assert loaded == obj
