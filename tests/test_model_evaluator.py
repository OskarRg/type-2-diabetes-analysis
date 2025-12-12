import json
import math
import numpy as np
import pytest
from pathlib import Path

# Assuming the class is located in source.model_evaluator
from source.model_evaluator import ModelEvaluator


@pytest.fixture
def evaluator():
    """Fixture to create a ModelEvaluator instance."""
    return ModelEvaluator()


@pytest.fixture
def dummy_data():
    """Fixture providing simple dummy X and y data."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    return X, y


def test_evaluate_with_proba_and_threshold(evaluator, dummy_data, mocker):
    """
    Test evaluation logic when the model supports predict_proba.
    Verifies that the threshold is correctly applied to probabilities.
    """
    X_test, y_test = dummy_data

    # 1. Mock the model using a list spec to explicitly allow both methods
    # We define that this mock ONLY has 'predict' and 'predict_proba' methods
    mock_model = mocker.Mock(spec=["predict", "predict_proba"])

    # Simulate that the model has predict_proba
    # Return probabilities for class 0 and 1.
    # Test case:
    # y_true = [0, 0, 1, 1]
    # proba_1 = [0.2, 0.4, 0.6, 0.8]
    # Standard predict (default threshold 0.5) would be [0, 0, 1, 1] -> Accuracy 1.0
    # BUT we set the threshold to 0.7!
    # Expected predictions: [0, 0, 0, 1] -> Error on the third element (is 1, predicted 0)
    mock_model.predict_proba.return_value = np.array(
        [[0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8]]
    )

    # Mock standard predict (should not be used for binary metrics here,
    # as the code in evaluate overrides y_pred based on the threshold)
    mock_model.predict.return_value = np.array([0, 0, 1, 1])

    # 2. Run evaluate with a high threshold (0.7)
    metrics = evaluator.evaluate(mock_model, X_test, y_test, threshold=0.7)

    # 3. Assertions
    # Check if predict_proba was called
    mock_model.predict_proba.assert_called_once_with(X_test)

    # Threshold logic check:
    # y_true: [0, 0, 1, 1]
    # proba:  [0.2, 0.4, 0.6, 0.8]
    # thresh 0.7 -> pred: [0, 0, 0, 1]
    # Result: 3 correct, 1 error. Accuracy = 0.75
    assert metrics["accuracy"] == 0.75

    # Check if ROC_AUC was calculated (is not NaN)
    assert not math.isnan(metrics["roc_auc"])
    # ROC AUC for perfectly sorted probabilities (0.2, 0.4 vs 0.6, 0.8) should be 1.0
    assert metrics["roc_auc"] == 1.0


def test_evaluate_without_predict_proba(evaluator, dummy_data, mocker):
    """
    Test evaluation logic when the model DOES NOT support predict_proba.
    Verifies fallback to standard .predict() and handling of NaN ROC_AUC.
    """
    X_test, y_test = dummy_data

    # 1. Mock the model with ONLY 'predict' in spec.
    # This ensures hasattr(mock_model, "predict_proba") returns False.
    mock_model = mocker.Mock(spec=["predict"])

    # Simulate y_pred (e.g., perfect predictions)
    mock_model.predict.return_value = np.array([0, 0, 1, 1])

    # 2. Run evaluate
    metrics = evaluator.evaluate(mock_model, X_test, y_test, threshold=0.5)

    # 3. Assertions
    mock_model.predict.assert_called_once_with(X_test)

    # Metrics should be perfect
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0

    # ROC AUC should be NaN because we don't have probabilities
    assert math.isnan(metrics["roc_auc"])


def test_evaluate_handles_zero_division(evaluator, dummy_data, mocker):
    """
    Test that precision/recall/f1 handle division by zero correctly
    (should return 0.0, not raise error).
    """
    X_test, y_test = dummy_data  # y_true = [0, 0, 1, 1]

    # Mock model with only 'predict'
    mock_model = mocker.Mock(spec=["predict"])

    # Model predicts only zeros -> TP=0, FP=0, FN=2, TN=2
    # Precision = TP / (TP + FP) = 0/0 -> ZeroDivision
    mock_model.predict.return_value = np.array([0, 0, 0, 0])

    metrics = evaluator.evaluate(mock_model, X_test, y_test, threshold=0.5)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0


def test_save_metrics_to_json(evaluator, tmp_path):
    """
    Test saving metrics to a JSON file.
    Uses tmp_path fixture for creating temporary files.
    """
    metrics = {"accuracy": 0.95, "f1": 0.94}
    # Create a path in a subdirectory to check if mkdir parents=True works
    output_file = tmp_path / "results" / "metrics.json"

    evaluator.save_metrics_to_json(metrics, output_file)

    assert output_file.exists()

    with open(output_file, "r") as f:
        loaded_metrics = json.load(f)

    assert loaded_metrics == metrics


def test_compare_models_prints_output(evaluator, capsys):
    """
    Test that compare_models prints the correct information to stdout.
    Uses capsys fixture to capture print statements.
    """
    metrics_map = {
        "Model_A": {"accuracy": 0.8, "roc_auc": 0.85},
        "Model_B": {"accuracy": 0.9, "roc_auc": 0.95},
    }

    evaluator.compare_models(metrics_map)

    captured = capsys.readouterr()
    output = captured.out

    # Check if key information is present in the output
    assert "===== Model_A =====" in output
    assert "===== Model_B =====" in output
    assert "accuracy: 0.800" in output
    assert "roc_auc: 0.950" in output
