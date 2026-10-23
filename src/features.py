"""
Feature engineering and preprocessing pipelines.
"""

from typing import List, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def build_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer handling numeric and categorical features.
    """
    transformers = []
    
    if numeric_features:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, numeric_features))
        
    if categorical_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, categorical_features))
        
    if not transformers:
        # Default fallback to standard scaler on all columns if not specified
        return ColumnTransformer(
            transformers=[("scaler", StandardScaler(), slice(None))],
            remainder="passthrough"
        )
        
    return ColumnTransformer(transformers=transformers, remainder="passthrough")
