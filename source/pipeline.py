import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

from source.data_loader import DataLoader
from source.feature_selector import FeatureSelector
from source.model_evaluator import ModelEvaluator
from source.model_trainer import LogisticRegressionStrategy, ModelTrainer, RandomForestStrategy
from source.preprocessor import Preprocessor
from source.utils import TARGET, LogisticRegressionParams, RandomForestParams
from source.visualizer import Visualizer

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from sklearn.base import ClassifierMixin

logger: logging.Logger = logging.getLogger(__name__)


class Pipeline:
    """Main class orchestrating the entire machine learning workflow."""

    def __init__(self) -> None:
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.feature_selector = FeatureSelector()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()
        self.visualizer = Visualizer()

    def run(self, data_path: str) -> None:
        """Execute the complete data preparation and modeling pipeline."""
        self.data_loader.load_data(data_path)
        self.data_loader.show_info()
        df: pd.DataFrame = self.data_loader.handle_missing()
        df = self.preprocessor.fix_categorical_values(df)
        df = self.preprocessor.encode_categorical(df)  # OHE should be used for SVM and LR
        df = self.preprocessor.normalize_features(df)
        #  TODO Add config with a section `correlation_based_features`
        correlation_info, best_features_list = self.feature_selector.correlation_analysis(
            df=df, target=TARGET
        )
        df = df[best_features_list + [TARGET]]
        # TODO Create DataSplitter class?
        from sklearn.model_selection import train_test_split

        X: pd.DataFrame = df.drop(columns=[TARGET])
        y: pd.Series = df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        # TODO This will be moved to config
        lr_params: LogisticRegressionParams = {
            "C": 1.0,
            "max_iter": 1000,
        }
        lr_strategy: LogisticRegressionStrategy = LogisticRegressionStrategy(params=lr_params)
        self.model_trainer.set_strategy(lr_strategy)
        lr_model: ClassifierMixin = self.model_trainer.train(X_train, y_train)
        # TODO  Implement ModelEvaluator - also include logging there.
        lr_pred: np.ndarray = lr_model.predict(X_test)
        lr_proba: np.ndarray = lr_model.predict_proba(X_test)[:, 1]

        logger.info(f"LR Accuracy: {accuracy_score(y_test, lr_pred):.3f}")
        logger.info(f"LR F1: {f1_score(y_test, lr_pred):.3f}")
        logger.info(f"LR ROC-AUC: {roc_auc_score(y_test, lr_proba):.3f}")
        logger.info(classification_report(y_test, lr_pred))

        lr_path: Path = Path("models") / "LR" / "model.joblib"
        self.model_trainer.save_model(lr_model, lr_path)

        rf_params: RandomForestParams = {
            "n_estimators": 200,
            "max_depth": None,
            "random_state": 42,
            "n_jobs": -1,
        }
        rf_strategy: RandomForestStrategy = RandomForestStrategy(params=rf_params)
        self.model_trainer.set_strategy(rf_strategy)
        rf_model: ClassifierMixin = self.model_trainer.train(X_train, y_train)

        rf_pred: np.ndarray = rf_model.predict(X_test)
        rf_proba: np.ndarray = rf_model.predict_proba(X_test)[:, 1]

        logger.info(f"LR Accuracy: {accuracy_score(y_test, rf_pred):.3f}")
        logger.info(f"LR F1: {f1_score(y_test, rf_pred):.3f}")
        logger.info(f"LR ROC-AUC: {roc_auc_score(y_test, rf_proba):.3f}")
        logger.info(classification_report(y_test, rf_pred))

        rf_path: Path = Path("models") / "RF" / "model.joblib"
        self.model_trainer.save_model(rf_model, rf_path)
