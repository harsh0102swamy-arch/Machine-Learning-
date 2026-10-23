"""
Inference script to make predictions using a trained model.

Usage:
    python predict.py --model models/random_forest_model.joblib --input data/raw/synthetic_sample.csv
"""

import argparse
from pathlib import Path
import pandas as pd
from src.model import load_model


def main():
    parser = argparse.ArgumentParser(description="Predict using a saved ML model.")
    parser.add_argument("--model", type=str, default="models/random_forest_model.joblib", help="Path to saved model.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV to predict on.")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output path for predictions.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at {input_path}")

    print(f"[INFO] Loading model from {args.model}...")
    model = load_model(args.model)

    print(f"[INFO] Loading input data from {input_path}...")
    df = pd.read_csv(input_path)
    # Remove target column if present in input
    features = df.drop(columns=["target"], errors="ignore")

    print(f"[INFO] Generating predictions for {len(features)} records...")
    preds = model.predict(features)
    
    result = features.copy()
    result["prediction"] = preds
    
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(features)
            for i in range(probas.shape[1]):
                result[f"probability_class_{i}"] = probas[:, i]
        except Exception:
            pass

    result.to_csv(args.output, index=False)
    print(f"[SUCCESS] Predictions saved to: {args.output}")
    print("\nPreview of predictions:")
    print(result.head())


if __name__ == "__main__":
    main()
