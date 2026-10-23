"""
Data loading and dataset generation utilities.
"""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def load_dataset_from_csv(file_path: str | Path, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads a dataset from CSV and splits it into features (X) and target (y).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {file_path}")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_informative: int = 6,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generates a reproducible synthetic classification dataset.
    Useful for demonstration, smoke tests, and template validation.
    """
    X_arr, y_arr = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        random_state=random_state,
    )
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    X = pd.DataFrame(X_arr, columns=feature_names)
    y = pd.Series(y_arr, name="target")
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits features and targets into train and test sets.
    """
    stratify_target = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )
