"""
Utility functions for logging, metrics visualization, and formatting.
"""

import json
from pathlib import Path
from typing import Dict, Any


def print_metrics(metrics: Dict[str, Any]) -> None:
    """
    Neatly prints evaluation metrics to stdout.
    """
    print("\n" + "=" * 45)
    print("           MODEL EVALUATION SUMMARY          ")
    print("=" * 45)
    print(f"  Accuracy:  {metrics.get('accuracy', 0.0):.4f}")
    print(f"  Precision: {metrics.get('precision', 0.0):.4f}")
    print(f"  Recall:    {metrics.get('recall', 0.0):.4f}")
    print(f"  F1-Score:  {metrics.get('f1', 0.0):.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC:   {metrics.get('roc_auc', 0.0):.4f}")
    print("=" * 45 + "\n")


def save_metrics_json(metrics: Dict[str, Any], filepath: str | Path) -> None:
    """
    Saves metrics dictionary to a JSON file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Metrics saved to {path}")
