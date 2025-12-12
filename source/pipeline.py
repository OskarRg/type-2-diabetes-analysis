import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from source.arguments_schema import (
    FeatureSelectionParams,
    LRExperiment,
    PipelineParams,
    RFExperiment,
)
from source.data_loader import DataLoader
from source.feature_selector import FeatureSelector
from source.model_evaluator import ModelEvaluator
from source.model_trainer import (
    BaseModelStrategy,
    LogisticRegressionStrategy,
    ModelTrainer,
    RandomForestStrategy,
)
from source.preprocessor import Preprocessor
from source.utils import (
    ExperimentResult,
    FixCategoricalStrategy,
    prepare_data_split,
)
from source.visualizer import Visualizer

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin

logger: logging.Logger = logging.getLogger(__name__)


class Pipeline:
    """Main class orchestrating the entire machine learning workflow."""

    def __init__(self, params: PipelineParams) -> None:
        self.params: PipelineParams = params
        self.data_loader: DataLoader = DataLoader()
        self.preprocessor: Preprocessor = Preprocessor()
        self.feature_selector: FeatureSelector = FeatureSelector()
        self.model_trainer: ModelTrainer = ModelTrainer()
        self.model_evaluator: ModelEvaluator = ModelEvaluator()
        self.visualizer: Visualizer = Visualizer()

    def run(self, data_path: str) -> None:
        """
        Execute the complete data preparation and modeling pipeline.

        :param data_path: Path to the dataset saved in a .csv format.
        """
        self.data_loader.load_data(data_path)
        self.data_loader.show_info()
        df: pd.DataFrame = self.data_loader.handle_missing()
        categorical_strategy: str = (
            FixCategoricalStrategy.DROP
            if self.params.data.preprocessing.drop_in_fix_categorical
            else FixCategoricalStrategy.CLIP
        )
        df = self.preprocessor.fix_categorical_values(df, strategy=categorical_strategy)
        if self.params.data.preprocessing.remove_outliers:
            df = self.preprocessor.remove_outliers(df)
        df = self.preprocessor.encode_categorical(df)
        df = self.preprocessor.normalize_features(df)

        target_col: str = self.params.data.target_column
        df = self._perform_feature_selection(df, target_col)

        X_train, X_test, y_train, y_test = prepare_data_split(df, target_col)

        timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plots_dir: Path = Path("results") / "plots" / timestamp
        plots_dir.mkdir(parents=True, exist_ok=True)
        all_results: list[dict] = []

        logger.info("--- Starting Logistic Regression Experiments ---")
        for exp in self.params.experiments.logistic_regression:
            result = self._run_single_experiment(
                experiment_config=exp,
                strategy_cls=LogisticRegressionStrategy,
                model_type_name="LogisticRegression",
                data=(X_train, y_train, X_test, y_test),
                plots_dir=plots_dir,
                timestamp=timestamp,
                advanced_plots=False,
            )
            all_results.append(result)

        logger.info("--- Starting Random Forest Experiments ---")
        for exp in self.params.experiments.random_forest:
            result = self._run_single_experiment(
                experiment_config=exp,
                strategy_cls=RandomForestStrategy,
                model_type_name="RandomForest",
                data=(X_train, y_train, X_test, y_test),
                plots_dir=plots_dir,
                timestamp=timestamp,
                advanced_plots=True,
            )
            all_results.append(result)

        self._save_results(all_results, timestamp)

    def _perform_feature_selection(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Handles correlation analysis and feature filtering logic.

        :param df: Input data DataFrame.
        :param target_col: Target column of the experiment.
        :return: Filtered DataFrame.
        """
        fs_params: FeatureSelectionParams = self.params.data.feature_selection

        if not fs_params.enabled:
            logger.info("Feature selection disabled. Using all features.")
            return df

        logger.info(f"Feature selection enabled. Method: {fs_params.method}")

        correlation_info: pd.Series = self.feature_selector.correlation_analysis(
            df=df, target=target_col
        )
        selected_features: list[str] = correlation_info.index.tolist()

        if fs_params.threshold is not None:
            logger.info(f"Filtering features with correlation < {fs_params.threshold}")
            valid_mask = correlation_info >= fs_params.threshold
            selected_features = correlation_info[valid_mask].index.tolist()
            logger.info(f"Features remaining after threshold: {len(selected_features)}")

        if fs_params.top_k is not None:
            if fs_params.top_k < len(selected_features):
                logger.info(
                    f"Selecting top {fs_params.top_k} features out of {len(selected_features)}"
                )
                selected_features = selected_features[: fs_params.top_k]
            else:
                logger.warning(
                    f"Top_k ({fs_params.top_k}) is larger than feature count. Keeping all."
                )

        logger.info(f"Final selected features: {selected_features}")
        return df[selected_features + [target_col]]

    def _run_single_experiment(
        self,
        experiment_config: LRExperiment | RFExperiment,
        strategy_cls: type,
        model_type_name: str,
        data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        plots_dir: Path,
        timestamp: str,
        advanced_plots: bool = False,
    ) -> dict:
        """
        This method encapsulates the full lifecycle of an experiment: initializing the
        strategy with specific parameters, training the model, evaluating it with a
        custom threshold, generating visualizations (Confusion Matrix, and optionally
        ROC), and aggregating all results into a dictionary.

        :param experiment_config: Configuration object containing model hyperparameters,
                                  experiment name, and decision threshold.
        :param strategy_cls: The class (type) of the modeling strategy to instantiate
                             (e.g., LogisticRegressionStrategy).
        :param model_type_name: Human-readable name of the model type for logging/results
                                (e.g., "RandomForest").
        :param data: A tuple containing the dataset split: (X_train, y_train, X_test, y_test).
        :param plots_dir: Path object pointing to the directory where plots should be saved.
        :param timestamp: String timestamp used to tag the result entry.
        :param advanced_plots: If True, generates additional plots like ROC curves.
        :return: A dictionary containing experiment metadata, metrics, paths to saved plots,
                 and the parameters used.
        """
        X_train, y_train, X_test, y_test = data
        exp_name: str = experiment_config.name

        logger.info(f"Running experiment: {exp_name}")

        params_dict: dict = experiment_config.params.model_dump()
        strategy: BaseModelStrategy = strategy_cls(params=params_dict)
        self.model_trainer.set_strategy(strategy)
        model: ClassifierMixin = self.model_trainer.train(X_train, y_train)

        current_threshold: float = experiment_config.threshold
        metrics: dict = self.model_evaluator.evaluate(
            model, X_test, y_test, threshold=current_threshold
        )
        logger.info(f"Results for {exp_name}: {metrics}")

        cm_path: Path = plots_dir / f"CM_{exp_name}.png"
        self.visualizer.save_confusion_matrix(
            model=model,
            X_test=X_test,
            y_test=y_test,
            save_path=cm_path,
            title=f"Confusion Matrix: {exp_name}",
            normalize=True,
            threshold=current_threshold,
        )

        roc_path_str: str | None = None
        if advanced_plots:
            roc_path: Path = plots_dir / f"ROC_{exp_name}.png"
            self.visualizer.save_roc_curve(
                model=model, X_test=X_test, y_test=y_test, save_path=roc_path
            )
            roc_path_str = str(roc_path)

        result_entry: ExperimentResult = {
            "experiment_name": exp_name,
            "model_type": model_type_name,
            "params": params_dict,
            "threshold": current_threshold,
            "metrics": metrics,
            "cm_plot_path": str(cm_path),
            "timestamp": timestamp,
        }

        if roc_path_str:
            result_entry["roc_plot_path"] = roc_path_str

        return result_entry

    def _save_results(self, results: list, timestamp: str) -> None:
        """
        Helper to save aggregated results to a JSON file.

        :param results: Final results for experiments.
        :param timestamp: Timestamp for the unique name creation.
        """
        results_dir: Path = Path("results")
        results_dir.mkdir(exist_ok=True)

        filename: Path = results_dir / f"experiments_run_{timestamp}.json"

        try:
            with open(filename, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"All experiment results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
