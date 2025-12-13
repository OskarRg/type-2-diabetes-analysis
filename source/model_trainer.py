from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from source.utils import LogisticRegressionParams, RandomForestParams


class BaseModelStrategy(ABC):
    """
    Base class for all model training strategies.
    Concrete subclasses must implement the `train` method.
    Concrete subclasses should be given adequate parameters.
    """

    @abstractmethod
    def get_estimator(self) -> ClassifierMixin:
        """
        Returns an unfitted model instance with configured parameters.
        This is useful for Cross-Validation or Sklearn Pipelines.
        """
        raise NotImplementedError

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
        """
        Train the model strategy on the provided data.

        :param X_train: Feature matrix used for training.
        :param y_train: Target labels used for training.

        :returns: A fitted scikit-learn classifier.
        """
        model: ClassifierMixin = self.get_estimator()
        model.fit(X_train, y_train)
        return model


class LogisticRegressionStrategy(BaseModelStrategy):
    """
    Logistic Regression training strategy.
    This strategy trains a model using `sklearn.linear_model.LogisticRegression`.
    """

    def __init__(self, params: LogisticRegressionParams) -> None:
        """
        Initializes passed parameters to the class.

        :param params: Arguments passed directly to `sklearn.linear_model.LogisticRegression`.
        """
        self.params: LogisticRegressionParams = params

    def get_estimator(self) -> ClassifierMixin:
        """
        Return an unfitted LogisticRegression classifier.

        :return: LogisticRegression classifier.
        """
        return LogisticRegression(**self.params)


class RandomForestStrategy(BaseModelStrategy):
    """
    Random Forest training strategy.
    This strategy trains a model using `sklearn.ensemble.RandomForestClassifier`.
    """

    def __init__(self, params: RandomForestParams) -> None:
        """
        Initializes passed parameters to the class.

        :param params: Arguments passed directly to `sklearn.ensemble.RandomForestClassifier`.
        """
        self.params: RandomForestParams = params

    def get_estimator(self) -> ClassifierMixin:
        """
        Return an unfitted RandomForestClassifier classifier.

        :return: RandomForestClassifier classifier.
        """
        return RandomForestClassifier(**self.params)


class XGBoostStrategy(BaseModelStrategy):
    """
    Initializes passed parameters to the class.

    :param params: Arguments passed directly to `xgboost.XGBClassifier`.
    """

    def __init__(self, params: dict) -> None:
        self.params = params

    def get_estimator(self) -> ClassifierMixin:
        """
        Return an unfitted XGBClassifier classifier.

        :return: XGBClassifier classifier.
        """
        return XGBClassifier(**self.params)


class ModelTrainer:
    """
    A trainer that delegates model training to a chosen strategy.
    """

    def __init__(self, strategy: BaseModelStrategy | None = None) -> None:
        """
        :param strategy: Initial strategy object. If not provided, it must
        be set later with `set_strategy`.
        """
        self.strategy: BaseModelStrategy | None = strategy

    def set_strategy(self, strategy: BaseModelStrategy) -> None:
        """
        Set the model training strategy.

        :param strategy: Strategy object defining how the model is trained.
        """
        self.strategy = strategy

    def get_estimator(self) -> ClassifierMixin:
        """
        Get unfitted estimator from current strategy.
        """
        if self.strategy is None:
            raise ValueError("No model strategy has been set.")
        return self.strategy.get_estimator()

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
        """
        Train the model using the current strategy.

        :param X_train: Feature matrix used for training.
        :param y_train: Target label used for training.
        :returns: The fitted model.
        :raises ValueError: If no strategy has been assigned.
        """
        if self.strategy is None:
            raise ValueError("No model strategy has been set.")
        return self.strategy.train(X_train, y_train)

    @staticmethod
    def save_model(model: ClassifierMixin, path: Path) -> None:
        """
        Save the trained model to a file using `joblib`.

        :param model: The trained model instance.
        :param path: Path where the model will be saved.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
