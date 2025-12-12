import pytest
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from source.model_trainer import (
    LogisticRegressionStrategy,
    RandomForestStrategy,
    ModelTrainer,
    BaseModelStrategy,
)

# --- Strategy Tests ---


def test_logistic_regression_strategy_train(sample_df):
    """Test if LogisticRegressionStrategy correctly trains a model."""
    # Arrange
    X = sample_df.drop(columns="Diabetes")
    y = sample_df["Diabetes"]

    params = {"C": 0.5, "max_iter": 100}
    strategy = LogisticRegressionStrategy(params)

    # Act
    model = strategy.train(X, y)

    # Assert
    assert isinstance(model, LogisticRegression)
    assert model.C == 0.5
    # Check if model is fitted (sklearn raises error if not)
    try:
        check_is_fitted(model)
    except NotFittedError:
        pytest.fail("LogisticRegression model was returned but not fitted.")


def test_random_forest_strategy_train(sample_df):
    """Test if RandomForestStrategy correctly trains a model."""
    # Arrange
    X = sample_df.drop(columns="Diabetes")
    y = sample_df["Diabetes"]

    params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}
    strategy = RandomForestStrategy(params)

    # Act
    model = strategy.train(X, y)

    # Assert
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 10
    # Check if model is fitted
    try:
        check_is_fitted(model)
    except NotFittedError:
        pytest.fail("RandomForest model was returned but not fitted.")


def test_base_model_strategy_is_abstract():
    """Test that BaseModelStrategy cannot be instantiated directly."""
    # BaseModelStrategy inherits from ABC and has an abstract method
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseModelStrategy()


# --- ModelTrainer Tests ---


def test_model_trainer_init_without_strategy():
    """Test Trainer initialization without a strategy."""
    trainer = ModelTrainer()
    assert trainer.strategy is None


def test_model_trainer_init_with_strategy():
    """Test Trainer initialization with a strategy."""
    strategy = LogisticRegressionStrategy({})
    trainer = ModelTrainer(strategy)
    assert trainer.strategy == strategy


def test_model_trainer_set_strategy():
    """Test setting a new strategy."""
    trainer = ModelTrainer()
    rf_strategy = RandomForestStrategy({})

    trainer.set_strategy(rf_strategy)

    assert trainer.strategy == rf_strategy
    assert isinstance(trainer.strategy, RandomForestStrategy)


def test_model_trainer_train_no_strategy_raises_error(sample_df):
    """Test that training without a strategy raises ValueError."""
    X = sample_df.drop(columns="Diabetes")
    y = sample_df["Diabetes"]

    trainer = ModelTrainer()  # strategy is None

    with pytest.raises(ValueError, match="No model strategy has been set"):
        trainer.train(X, y)


def test_model_trainer_delegates_training(sample_df, mocker):
    """Test that trainer delegates the train call to the strategy."""
    # Arrange
    X = sample_df.drop(columns="Diabetes")
    y = sample_df["Diabetes"]

    # Mock the strategy to check if its train method was called
    # Use autospec=True so the mock behaves like a class instance
    mock_strategy = mocker.create_autospec(BaseModelStrategy, instance=True)
    # The mock must return a ClassifierMixin object (or a mock pretending to be one)
    mock_model = mocker.Mock(spec=ClassifierMixin)
    mock_strategy.train.return_value = mock_model

    trainer = ModelTrainer(strategy=mock_strategy)

    # Act
    result_model = trainer.train(X, y)

    # Assert
    mock_strategy.train.assert_called_once_with(X, y)
    assert result_model == mock_model


def test_save_model_creates_directory_and_saves(tmp_path, mocker):
    """
    Test saving the model.
    Uses tmp_path (pytest fixture) for temporary directories.
    Uses mocker to verify joblib.dump was called without actually writing a heavy object.
    """
    # Arrange
    # Create a dummy model
    mock_model = mocker.Mock(spec=ClassifierMixin)

    # Path in a non-existent folder (to test mkdir)
    save_path = tmp_path / "subdir" / "models" / "model.joblib"

    # Mock joblib.dump to avoid creating a physical file (unless desired)
    mock_joblib_dump = mocker.patch("joblib.dump")

    # Act
    ModelTrainer.save_model(mock_model, save_path)

    # Assert
    # 1. Check if directories were created (path.parent.mkdir)
    assert save_path.parent.exists()
    assert save_path.parent.is_dir()

    # 2. Check if joblib.dump was called with correct arguments
    mock_joblib_dump.assert_called_once_with(mock_model, save_path)


# Optional: Integration test for saving (without mocking joblib),
# to ensure it physically saves the file.
def test_save_model_integration(tmp_path):
    """Integration test ensuring a real file is created."""
    model = LogisticRegression()

    X_dummy = [[0], [1]]
    y_dummy = [0, 1]

    model.fit(X_dummy, y_dummy)

    save_path = tmp_path / "real_model.joblib"

    ModelTrainer.save_model(model, save_path)

    assert save_path.exists()
    assert save_path.is_file()
