"""
Main training pipeline script.

Usage:
    # Run with default synthetic dataset:
    python train.py

    # Or specify custom CSV:
    python train.py --csv data/raw/dataset.csv --target label
"""

import argparse
from pathlib import Path
from src.data_loader import generate_synthetic_data, load_dataset_from_csv, split_data
from src.features import build_preprocessor
from src.model import create_pipeline, evaluate_model, save_model
from src.utils import print_metrics, save_metrics_json


def main():
    parser = argparse.ArgumentParser(description="Train a Machine Learning classification model.")
    parser.add_argument("--csv", type=str, default=None, help="Path to input CSV dataset.")
    parser.add_argument("--target", type=str, default="target", help="Target column name in CSV.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for test set.")
    parser.add_argument("--output-model", type=str, default="models/random_forest_model.joblib", help="Output model path.")
    parser.add_argument("--output-metrics", type=str, default="models/metrics.json", help="Output metrics path.")
    args = parser.parse_args()

    print("[1/5] Loading dataset...")
    if args.csv:
        X, y = load_dataset_from_csv(args.csv, target_column=args.target)
        print(f"Loaded {len(X)} rows from {args.csv}")
    else:
        print("No CSV supplied; generating demonstration dataset (1,000 samples, 10 features)...")
        X, y = generate_synthetic_data(n_samples=1000, n_features=10, random_state=42)
        # Save a sample to data/raw for demonstration
        raw_demo_path = Path("data/raw/synthetic_sample.csv")
        raw_demo_path.parent.mkdir(parents=True, exist_ok=True)
        demo_df = X.copy()
        demo_df["target"] = y
        demo_df.to_csv(raw_demo_path, index=False)
        print(f"Sample dataset saved to {raw_demo_path}")

    print("[2/5] Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=args.test_size, random_state=42)
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    print("[3/5] Building preprocessing pipeline & model...")
    preprocessor = build_preprocessor(numeric_features=list(X.columns))
    pipeline = create_pipeline(preprocessor=preprocessor, random_state=42)

    print("[4/5] Training model...")
    pipeline.fit(X_train, y_train)

    print("[5/5] Evaluating model performance...")
    metrics = evaluate_model(pipeline, X_test, y_test)
    print_metrics(metrics)

    # Persist artifacts
    save_model(pipeline, args.output_model)
    save_metrics_json(metrics, args.output_metrics)
    print("[SUCCESS] Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
