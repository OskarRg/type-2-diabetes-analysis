import json
import logging
from pathlib import Path

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from source.utils import MetricsDict

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates and compares machine learning model performance.

    This class provides:

    - computation of common classification metrics
    - confusion matrix visualization
    - comparison of multiple models based on their metrics
    """

    def __init__(self) -> None:
        """Initialize the evaluator."""
        pass

    @staticmethod
    def evaluate(
        model: ClassifierMixin,
        X_test: np.ndarray | list[list[float]],
        y_test: np.ndarray | list[int],
        threshold: float,
    ) -> MetricsDict:
        """
        Evaluate a model and compute standard classification metrics.

        :param model: Trained classifier implementing `predict` and `predict_proba`.
        :param X_test: Test feature matrix.
        :param y_test: Ground-truth target values.
        :param json_file: File where a JSON report should be saved.
        :returns: Dictionary containing computed evaluation metrics.
        """

        logger.info("Evaluating model performance...")

        y_proba: np.ndarray | None = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X_test)

        metrics: MetricsDict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
        }

        metrics["roc_auc"] = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")

        print("Evaluation metrics:")
        for key, value in metrics.items():
            print(f"{key.upper()}: {value:.3f}")

        print("Classification Report:")
        print("\n" + classification_report(y_test, y_pred, zero_division=0))

        return metrics

    @staticmethod
    def save_metrics_to_json(metrics: MetricsDict, json_path: Path) -> None:
        """
        Save evaluation metrics to a JSON file.

        :param metrics: Dictionary containing model metrics.
        :param path: Destination file path.
        """

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

    @staticmethod
    def compare_models(metrics_dict_by_model: dict[str, MetricsDict]) -> None:
        """
        Compare multiple models by printing their metrics side-by-side.

        :param metrics_dict: A dictionary where keys are model names
                             and values are metric dictionaries.
        Example:
            {
                "LogisticRegression": {"accuracy": 0.84, "f1": 0.81},
                "RandomForest": {"accuracy": 0.89, "f1": 0.86},
            }
        """
        # TODO Optionally add coloring

        logger.info("Comparing models...")

        for model_name, metrics in metrics_dict_by_model.items():
            print(f"\n===== {model_name} =====")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.3f}")
