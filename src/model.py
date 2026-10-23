"""
Model training, evaluation, and serialization functions.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.pipeline import Pipeline


def create_pipeline(
    preprocessor: Any,
    classifier: Optional[BaseEstimator] = None,
    random_state: int = 42
) -> Pipeline:
    """
    Creates an end-to-end pipeline combining preprocessing and estimator.
    """
    if classifier is None:
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=random_state
        )
        
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluates model performance and returns metrics.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    # Try ROC AUC if predict_proba is supported
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
        except Exception:
            pass
            
    return metrics


def save_model(model: Pipeline, filepath: str | Path) -> None:
    """
    Serializes model to disk using joblib.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[INFO] Model successfully saved to: {path}")


def load_model(filepath: str | Path) -> Pipeline:
    """
    Loads serialized model pipeline from disk.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)
