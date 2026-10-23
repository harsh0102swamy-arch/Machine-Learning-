"""
Unit tests for data loader, feature transformations, model pipeline, and inference.
"""

from pathlib import Path
import pytest
from src.data_loader import generate_synthetic_data, split_data
from src.features import build_preprocessor
from src.model import create_pipeline, evaluate_model, save_model, load_model


def test_synthetic_data_generation():
    X, y = generate_synthetic_data(n_samples=100, n_features=5, random_state=42)
    assert len(X) == 100
    assert len(y) == 100
    assert X.shape[1] == 5


def test_split_data():
    X, y = generate_synthetic_data(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=42)
    assert len(X_train) == 75
    assert len(X_test) == 25


def test_pipeline_fit_evaluate_save_load(tmp_path):
    X, y = generate_synthetic_data(n_samples=200, n_features=6, random_state=42)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor(numeric_features=list(X.columns))
    model = create_pipeline(preprocessor=preprocessor, random_state=42)

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert metrics["accuracy"] >= 0.5  # reasonable baseline on synthetic data

    # Save and Load
    saved_model_path = tmp_path / "test_model.joblib"
    save_model(model, saved_model_path)
    assert saved_model_path.exists()

    loaded_model = load_model(saved_model_path)
    preds = loaded_model.predict(X_test)
    assert len(preds) == len(X_test)
